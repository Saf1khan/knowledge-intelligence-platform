# Memory & Context Guide - Knowledge Intelligence Platform

## Project Overview
**Knowledge Intelligence Platform** is an enterprise-grade AI/RAG platform designed for processing, chunking, embedding, indexing, retrieving, and evaluating complex knowledge documents (such as PDFs).

- **Current Branch**: `feature/backend-foundation`
- **Primary Backend Stack**: Python 3.x, FastAPI, pypdf, tiktoken, NumPy, Scikit-learn (planned), Vector DB (Qdrant/Milvus planned)
- **Primary Frontend Stack**: Next.js / TypeScript (planned in `frontend/`)
- **Virtual Environment Path**: `backend/venv/`

---

## Progress So Far

### Completed Work
1. **Backend Infrastructure Setup**:
   - Initialized FastAPI server (`backend/app/main.py`) with `/` and `/health` endpoints.
   - Configured repository `.gitignore`, VSCode settings, and `pyproject.toml`.
   - Setup `backend/requirements.txt` with essential libraries (`fastapi`, `uvicorn`, `pypdf`, `tiktoken`, `numpy`, `scikit-learn`, `python-dotenv`, etc.).

2. **Ingestion & Processing Services (`backend/app/services/ingestion/`)**:
   - **`pdf_parser.py`**: Robust PDF text extraction service using `pypdf` with error handling, empty string checks, encryption fallback, and metrics collection (`PDFDocument` dataclass).
   - **`chunking.py`**: Enterprise-grade document chunking service implementing sliding window, sentence-aware chunking, token counting (`tiktoken`), semantic overlap preservation, and metadata enrichment.

---

## Status & Remaining Work Roadmap

### Phase 1: Ingestion & Document Processing Pipelines (In Progress)
- [x] PDF Parsing & Extraction Service
- [x] Text Chunking Service (Sliding Window & Token-Based)
- [ ] API routes for file upload & ingestion (`backend/app/api/v1/ingestion.py`)
- [ ] Support for additional document types (Markdown, DOCX, TXT)
- [ ] Async background processing queue (e.g. Celery / ARQ / FastAPI BackgroundTasks)

### Phase 2: Embeddings & Vector Indexing
- [ ] Embedding generation service (`backend/app/services/embeddings/`) supporting OpenAI/HuggingFace models
- [ ] Vector Database integration (Qdrant / Chroma / Weaviate / PGVector)
- [ ] Index management & metadata storage schema

### Phase 3: Hybrid Retrieval & Search Service
- [ ] Semantic vector search
- [ ] Sparse BM25 / Keyword search implementation
- [ ] Hybrid re-ranking mechanism (Cross-Encoder / Reciprocal Rank Fusion)

### Phase 4: LLM & RAG Agent Services
- [ ] Prompt management & context builder (`backend/app/services/llm/`)
- [ ] RAG Agent execution & citation tracking (`backend/app/services/agents/`)
- [ ] Conversational memory management

### Phase 5: Evaluation & Analytics Pipeline
- [ ] Retrieval evaluation metrics (MRR, NDGG, Precision@K) (`backend/app/services/evaluation/`)
- [ ] RAG output quality assessment (Faithfulness, Answer Relevance)

### Phase 6: Frontend Interface
- [ ] Next.js dashboard for document upload, search UI, chat interface, and evaluation metrics visualization

---

## Suggested Tasks For Today

1. **Task A: Build Ingestion API Endpoint & Unit Tests (Recommended)**
   - Create FastAPI endpoints for file uploads (`POST /api/v1/ingest/pdf`) in `backend/app/api/v1/ingestion.py`.
   - Write comprehensive pytest unit tests for `pdf_parser.py` and `chunking.py`.

2. **Task B: Implement Embeddings Service**
   - Build `backend/app/services/embeddings/` using SentenceTransformers / OpenAI API to convert chunked text into vector representations.

3. **Task C: Integrate Vector DB / Persistence**
   - Set up local Qdrant/Chroma vector DB integration for indexing chunks with metadata.

---

## How to Resume Work Next Time
1. Check `MEMORY.md` for current project state.
2. Activate virtual environment: `backend\venv\Scripts\activate`
3. Launch backend API: `uvicorn backend.app.main:app --reload`
