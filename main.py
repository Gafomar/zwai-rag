from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from typing import List, Optional
import PyPDF2
import docx
import io
import httpx
import uuid
import hmac
import socket
import ipaddress
from urllib.parse import urlparse
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny

app = FastAPI(title="ZWAI RAG Service")

# Configuration
# CORS_ORIGINS: comma-separated list of allowed origins (empty = no browser origins allowed;
# this is a server-to-server service).
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
# RAG_API_KEY: when set, every endpoint except / and /health requires "X-API-Key".
RAG_API_KEY = os.getenv("RAG_API_KEY")
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "20"))  # cost control
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(20 * 1024 * 1024)))
EMBEDDING_MODEL = "text-embedding-3-small"

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Optional[str] = Header(None)):
    if not RAG_API_KEY:
        return  # auth disabled (local/dev)
    if not x_api_key or not hmac.compare_digest(x_api_key, RAG_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# OpenAI client factory (lazy initialization)
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COLLECTION_NAME = "knowledge_base"

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    company_id: str
    accessible_doc_ids: List[str]
    limit: int = Field(5, ge=1, le=50)

class DeleteRequest(BaseModel):
    document_id: str
    company_id: str

# Initialize Qdrant collection on startup
@app.on_event("startup")
async def startup():
    if qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists")
    else:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"Created collection '{COLLECTION_NAME}'")
    # Payload indexes keep the tenant/document filters fast
    for field in ("company_id", "document_id"):
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema="keyword",
        )

# Health check endpoint
@app.get("/")
def root():
    return {"service": "ZWAI RAG Service", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "zwai-rag"}

# Extract text from different file types
def extract_text(file_bytes: bytes, filename: str) -> str:
    try:
        if filename.lower().endswith('.pdf'):
            pdf = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
            return text
        
        elif filename.lower().endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        elif filename.lower().endswith(('.txt', '.md')):
            return file_bytes.decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    
    except Exception as e:
        raise ValueError(f"Text extraction failed: {str(e)}")

def chunk_text(text: str, size: int = 1000, step: int = 800) -> List[str]:
    """Split text into overlapping chunks (size 1000, overlap 200 by default)."""
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + size]
        if len(chunk) > 50:  # Skip tiny chunks
            chunks.append(chunk)
    return chunks


def validate_public_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts resolving to non-public addresses (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("file_url must be an http(s) URL")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        raise ValueError("file_url host could not be resolved")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError("file_url must point to a public host")


async def download_file(url: str) -> bytes:
    validate_public_url(url)
    data = bytearray()
    # Redirects are not followed: a redirect could point at an internal address.
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async for part in response.aiter_bytes():
                data.extend(part)
                if len(data) > MAX_FILE_BYTES:
                    raise ValueError(f"File exceeds {MAX_FILE_BYTES} bytes")
    return bytes(data)


def delete_document_points(document_id: str, company_id: str) -> None:
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="company_id", match=MatchValue(value=company_id)),
            ]
        ),
    )


# Index document endpoint - ACCEPTS FILE OR FILE_URL
@app.post("/index", dependencies=[Depends(require_api_key)])
async def index_document(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    document_id: str = Form(...),
    company_id: str = Form(...)
):
    try:
        print(f"Indexing document ID: {document_id}")
        
        # Get file bytes - either from upload or download from URL
        if file:
            file_bytes = await file.read(MAX_FILE_BYTES + 1)
            if len(file_bytes) > MAX_FILE_BYTES:
                raise ValueError(f"File exceeds {MAX_FILE_BYTES} bytes")
            filename = file.filename
            print(f"File uploaded: {filename}, size: {len(file_bytes)} bytes")
            
        elif file_url:
            print(f"Downloading file from: {file_url}")
            file_bytes = await download_file(file_url)
            # Extract filename from URL (remove query params)
            filename = file_url.split('/')[-1].split('?')[0]
            if not filename or '.' not in filename:
                filename = 'document.pdf'  # Default
            print(f"Downloaded: {filename}, size: {len(file_bytes)} bytes")
            
        else:
            raise ValueError("Either 'file' or 'file_url' must be provided")
        
        # Extract text
        text = await run_in_threadpool(extract_text, file_bytes, filename)
        print(f"Extracted text: {len(text)} characters")
        
        if len(text) < 50:
            raise ValueError("Insufficient text extracted from document")
        
        # Create chunks (1000 chars, 200 overlap)
        chunks = chunk_text(text)
        
        print(f"Created {len(chunks)} chunks")
        
        # Limit chunks (cost control)
        chunks = chunks[:MAX_CHUNKS]
        
        # Generate embeddings
        print("Generating embeddings with OpenAI...")
        openai = get_openai_client()
        embeddings_response = await run_in_threadpool(
            openai.embeddings.create, model=EMBEDDING_MODEL, input=chunks
        )
        
        print(f"Generated {len(embeddings_response.data)} embeddings")
        
        # Prepare points for Qdrant - USE UUID FOR IDs
        points = []
        for idx, (chunk, embedding_data) in enumerate(zip(chunks, embeddings_response.data)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding_data.embedding,
                payload={
                    "company_id": company_id,
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_text": chunk,
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                }
            ))
        
        # Re-indexing replaces the previous version instead of duplicating chunks
        await run_in_threadpool(delete_document_points, document_id, company_id)

        # Store in Qdrant
        print(f"Storing {len(points)} points in Qdrant...")
        await run_in_threadpool(
            qdrant_client.upsert,
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )
        
        print(f"Successfully indexed document {document_id}")
        
        return {
            "success": True,
            "chunks_indexed": len(points),
            "text_length": len(text),
            "document_id": document_id
        }
        
    except Exception as e:
        print(f"Error indexing document: {str(e)}")
        if isinstance(e, (ValueError, httpx.HTTPError)):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Search documents endpoint
@app.post("/search", dependencies=[Depends(require_api_key)])
async def search_documents(request: SearchRequest):
    try:
        print(f"Searching for: {request.query}")
        print(f"Company: {request.company_id}, Accessible docs: {len(request.accessible_doc_ids)}")
        
        # Access control: no accessible documents means no results
        if not request.accessible_doc_ids:
            return {"success": True, "results": []}
        
        # Generate query embedding
        openai = get_openai_client()
        embedding_response = await run_in_threadpool(
            openai.embeddings.create, model=EMBEDDING_MODEL, input=request.query
        )
        query_vector = embedding_response.data[0].embedding
        
        query_filter = Filter(
            must=[
                FieldCondition(key="company_id", match=MatchValue(value=request.company_id)),
                FieldCondition(key="document_id", match=MatchAny(any=request.accessible_doc_ids)),
            ]
        )
        
        # Search Qdrant
        results = await run_in_threadpool(
            qdrant_client.search,
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=request.limit,
            score_threshold=0.7,
        )
        
        print(f"Found {len(results)} results")
        
        return {
            "success": True,
            "results": [
                {
                    "document_id": r.payload["document_id"],
                    "filename": r.payload["filename"],
                    "chunk_text": r.payload["chunk_text"],
                    "chunk_index": r.payload.get("chunk_index", 0),
                    "score": r.score
                }
                for r in results
            ]
        }
        
    except Exception as e:
        print(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Delete document endpoint
@app.post("/delete", dependencies=[Depends(require_api_key)])
async def delete_document(request: DeleteRequest):
    try:
        print(f"Deleting document: {request.document_id}")
        
        # Delete all points for this document using proper Filter syntax
        await run_in_threadpool(delete_document_points, request.document_id, request.company_id)
        
        print(f"Successfully deleted document {request.document_id}")
        
        return {
            "success": True,
            "deleted": request.document_id
        }
        
    except Exception as e:
        print(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoint
@app.get("/test", dependencies=[Depends(require_api_key)])
def test():
    return {
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "qdrant_configured": bool(os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY")),
        "collection": COLLECTION_NAME
    }
