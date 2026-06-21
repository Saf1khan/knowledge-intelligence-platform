from fastapi import FastAPI

app = FastAPI(
    title="Knowledge Intelligence Platform",
    version="1.0.0"
)

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
