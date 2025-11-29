"""
Utilities for Healthcare Classification System

This module contains shared constants and utilities for the healthcare
classification system.
"""

import torch
from datetime import datetime
from pathlib import Path

# Categories for medical vs insurance classification
CATEGORIES: list[str] = ["medical", "insurance"]

# Model and training configuration
MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"
CHECKPOINT_PATH = "classifier/checkpoints"
DATETIME_FORMAT = "%Y%m%d_%H%M%S"

# Device configuration
def get_device():
    """Get the best available device for training/inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()

def get_models():
    """
    Get the embedding model and classifier head for inference.
    
    Returns:
        tuple: (embedding_model, classifier_head)
    """
    from sentence_transformers import SentenceTransformer
    from head import ClassifierHead
    
    # Load embedding model
    embedding_model = SentenceTransformer(
        MODEL_NAME,
        prompts={
            'classification': 'task: classification | query: ',
            'retrieval (query)': 'task: search result | query: ',
            'retrieval (document)': 'title: {title | "none"} | text: ',
        },
        default_prompt_name='classification',
    )
    
    # Load classifier head (for 2 categories: medical, insurance)
    classifier_head = ClassifierHead(len(CATEGORIES))
    
    return embedding_model.to(DEVICE), classifier_head.to(DEVICE)

def get_timestamp():
    """Get current timestamp in standard format."""
    return datetime.now().strftime(DATETIME_FORMAT)

def ensure_checkpoint_dir():
    """Ensure checkpoint directory exists."""
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)