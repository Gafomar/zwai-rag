from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List
import PyPDF2
import docx
import io
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

app = FastAPI(title="ZWAI RAG Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "knowledge_base"

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    company_id: str
    accessible_doc_ids: List[str]
    limit: int = 5

class DeleteRequest(BaseModel):
    document_id: str
    company_id: str

# Initialize Qdrant collection on startup
@app.on_event("startup")
async def startup():
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists")
    except:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"Created collection '{COLLECTION_NAME}'")

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
                text += page.extract_text() + "\n"
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

# Index document endpoint
@app.post("/index")
async def index_document(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    company_id: str = Form(...)
):
    try:
        print(f"Indexing document: {file.filename} (ID: {document_id})")
        
        # Read file
        file_bytes = await file.read()
        print(f"File size: {len(file_bytes)} bytes")
        
        # Extract text
        text = extract_text(file_bytes, file.filename)
        print(f"Extracted text: {len(text)} characters")
        
        if len(text) < 50:
            raise ValueError("Insufficient text extracted from document")
        
        # Create chunks (1000 chars, 200 overlap)
        chunks = []
        for i in range(0, len(text), 800):
            chunk = text[i:i+1000]
            if len(chunk) > 50:  # Skip tiny chunks
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks")
        
        # Limit to 20 chunks for now (cost control)
        chunks = chunks[:20]
        
        # Generate embeddings
        print("Generating embeddings with OpenAI...")
        embeddings_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunks
        )
        
        print(f"Generated {len(embeddings_response.data)} embeddings")
        
        # Prepare points for Qdrant
        points = []
        for idx, (chunk, embedding_data) in enumerate(zip(chunks, embeddings_response.data)):
            points.append(PointStruct(
                id=f"{document_id}_{idx}",
                vector=embedding_data.embedding,
                payload={
                    "company_id": company_id,
                    "document_id": document_id,
                    "filename": file.filename,
                    "chunk_text": chunk,
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                }
            ))
        
        # Store in Qdrant
        print(f"Storing {len(points)} points in Qdrant...")
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
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
        raise HTTPException(status_code=500, detail=str(e))

# Search documents endpoint
@app.post("/search")
async def search_documents(request: SearchRequest):
    try:
        print(f"Searching for: {request.query}")
        print(f"Company: {request.company_id}, Accessible docs: {len(request.accessible_doc_ids)}")
        
        # Generate query embedding
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=request.query
        )
        
        query_vector = embedding_response.data[0].embedding
        
        # Build filter
        filter_conditions = {
            "must": [
                {"key": "company_id", "match": {"value": request.company_id}}
            ]
        }
        
        # Add document filter if provided
        if request.accessible_doc_ids:
            filter_conditions["must"].append({
                "key": "document_id",
                "match": {"any": request.accessible_doc_ids}
            })
        
        # Search Qdrant
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=filter_conditions,
            limit=request.limit,
            score_threshold=0.7
        )
        
        print(f"Found {len(results)} result
