"""
Utilities for Healthcare Classification System

This module contains shared constants and utilities for the healthcare
classification system.
"""

from classifier.head import ClassifierHead

import os
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
from pathlib import Path

MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"
CLASSIFIER_NAME = "davidgray/health-query-triage"
CATEGORIES: list[str] = ["medical", "insurance"]

# Model and training configuration
MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"
CHECKPOINT_PATH = "classifier/checkpoints"
DATETIME_FORMAT = "%Y%m%d_%H%M%S"

# Device configuration - use David's newer approach with fallback
try:
    DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
except AttributeError:
    # Fallback for older PyTorch versions
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

print(f"Using {DEVICE} device")

def get_models(model_id: str | None = None, num_labels: int = len(CATEGORIES)) -> tuple[SentenceTransformer, ClassifierHead]:
    """
    Loads embeddinggemma-300m-medical model and initializes the classification head.
    
    Returns:
        tuple: (embedding_model, classifier_head)
    """
    try:
        model_body = SentenceTransformer(
            MODEL_NAME,
            prompts={
                'classification': 'task: classification | query: ',
                'retrieval (query)': 'task: search result | query: ',
                'retrieval (document)': 'title: {title | "none"} | text: ',
            },
            default_prompt_name='classification',
        )

        if model_id:
            model_head = ClassifierHead.from_pretrained(model_id)
        else:
            model_head = ClassifierHead(num_labels)

    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Please ensure you have an internet connection and the transformers library installed.")
        raise RuntimeError("Failed to load the embedding model.")
    
    return model_body.to(DEVICE), model_head.to(DEVICE)

def get_latest_checkpoint(checkpoint_path: str):
    return os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path))[-1])
