import os
from pathlib import Path
from typing import Dict, List
import torch
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model Configuration
    MODEL_NAME: str = "sentence-transformers/embeddinggemma-300m-medical"
    CLASSIFIER_NAME: str = "davidgray/health-query-triage"
    CATEGORIES: List[str] = ["medical", "insurance"]
    
    # Paths
    CHECKPOINT_PATH: str = "classifier/checkpoints"
    CACHE_DIR: str = ".cache/embeddings"
    
    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Corpora Configuration
    CORPORA_CONFIG: Dict[str, dict] = {
        "medical_qa": {"path": "data/corpora/medical_qa.jsonl",
                       "text_fields": ["question", "answer", "title"]},
        "miriad":     {"path": "data/corpora/miriad_text.jsonl",
                       "text_fields": ["question", "answer", "title"]},
        "unidoc":     {"path": "data/corpora/unidoc_qa.jsonl",
                       "text_fields": ["question", "answer", "title"]},
    }

    class Config:
        env_file = ".env"

settings = Settings()
