from fastapi import FastAPI
from backend.app.api.upload import router as upload_router

app = FastAPI(
    title="Knowledge Intelligence Platform",
    version="1.0.0"
)

app.include_router(upload_router, prefix="/api/v1", tags=["Ingestion"])

@app.get("/")
def root():
    return {
        "message": "Knowledge Intelligence Platform API"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
