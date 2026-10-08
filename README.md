# zwai-rag
External RAG service for multi-tenant enterprise AI assistant

## Configuration

| Variable | Purpose |
|---|---|
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant connection |
| `OPENAI_API_KEY` | Embeddings |
| `RAG_API_KEY` | When set, all endpoints except `/` and `/health` require an `X-API-Key` header |
| `CORS_ORIGINS` | Comma-separated allowed browser origins (default: none) |
| `MAX_CHUNKS` | Max chunks indexed per document (default 20) |
| `MAX_FILE_BYTES` | Max upload/download size (default 20 MB) |

## Tests

```
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
