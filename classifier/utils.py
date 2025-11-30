"""
Utilities for Healthcare Classification System

This module contains shared constants and utilities for the healthcare
classification system.
"""

from classifier.head import ClassifierHead

import datasets as ds
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
from pathlib import Path

# Categories for medical vs insurance classification
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

def get_models(num_labels: int = len(CATEGORIES)) -> tuple[SentenceTransformer, ClassifierHead]:
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

        model_head = ClassifierHead(num_labels)

    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Please ensure you have an internet connection and the transformers library installed.")
        raise RuntimeError("Failed to load the embedding model.")
    
    return model_body.to(DEVICE), model_head.to(DEVICE)

def get_timestamp():
    """Get current timestamp in standard format."""
    return datetime.now().strftime(DATETIME_FORMAT)

def ensure_checkpoint_dir():
    """Ensure checkpoint directory exists."""
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
