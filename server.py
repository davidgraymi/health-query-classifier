import contextlib
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from anyio import to_thread

from pipeline import HealthQueryPipeline

# Global pipeline instance
pipeline = HealthQueryPipeline(use_reranker=False)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("Server starting up, loading models...")
    # We run initialization in a thread to avoid blocking the event loop
    await to_thread.run_sync(pipeline.initialize)
    yield
    print("Server shutting down...")

app = FastAPI(title="Health Query Classifier API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    k: int = 10

class RetrievalHit(BaseModel):
    id: str
    title: str
    text: str
    meta: Dict[str, Any]
    bm25: float
    dense: float
    rrf: float

class ClassificationResult(BaseModel):
    prediction: str
    probabilities: Dict[str, float]

class QueryResponse(BaseModel):
    query: str
    classification: ClassificationResult
    retrieval: List[RetrievalHit]

@app.post("/predict", response_model=QueryResponse)
async def predict(request: QueryRequest):
    try:
        # Run the CPU/GPU-bound inference in a separate thread
        result = await to_thread.run_sync(pipeline.predict, request.query, request.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "initialized": pipeline.is_initialized}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
