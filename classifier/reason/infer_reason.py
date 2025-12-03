"""
Inference module for Healthcare Reason Classification

This module provides inference for the reason classification system,
separate from the medical/insurance classifier.
"""

from ..head import ClassifierHead
from datetime import datetime
import os
import pprint
import torch
from sentence_transformers import SentenceTransformer

# Reason-specific configuration
REASON_CATEGORIES = {
    0: "ROUTINE_CARE",
    1: "PAIN_CONDITIONS", 
    2: "INJURIES",
    3: "SKIN_CONDITIONS",
    4: "STRUCTURAL_ISSUES",
    5: "PROCEDURES"
}

REASON_CHECKPOINT_PATH = "classifier/reason_checkpoints"
DATETIME_FORMAT = "%Y%m%d_%H%M%S"
MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"

def get_device():
    """Get the best available device for inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()

def get_reason_models():
    """Get the embedding model and classifier head for reason inference."""
    # Load embedding model
    embedding_model = SentenceTransformer(
        MODEL_NAME,
        prompts={
            'classification': 'task: healthcare reason classification | query: ',
            'retrieval (query)': 'task: search result | query: ',
            'retrieval (document)': 'title: {title | "none"} | text: ',
        },
        default_prompt_name='classification',
    )
    
    # Load classifier head (for 6 reason categories)
    classifier_head = ClassifierHead(len(REASON_CATEGORIES))
    
    return embedding_model.to(DEVICE), classifier_head.to(DEVICE)

def predict_reason_query(
    text: list[str],
    embedding_model: SentenceTransformer,
    classifier_head: ClassifierHead,
) -> dict:
    """
    Runs the full inference pipeline for reason classification: Text -> Embedding -> Classification.
    """
    # Set models to evaluation mode
    embedding_model.eval()
    classifier_head.eval()

    with torch.no_grad():
        # Embed the text
        embeddings = embedding_model.encode(
            text,
            convert_to_tensor=True,
            device=DEVICE
        ).to(DEVICE)

        # Calculate probabilities and prediction
        probabilities = classifier_head.predict_proba(embeddings)

        # Get the predicted index and confidence
        predicted_indices = torch.argmax(probabilities, dim=1)
        
        # Convert tensors to Python types safely
        if predicted_indices.dim() == 0:  # Single prediction
            predicted_indices = [predicted_indices.item()]
        else:
            predicted_indices = predicted_indices.cpu().tolist()
        
        # Get confidences
        confidences = []
        for i, idx in enumerate(predicted_indices):
            conf = probabilities[i][idx].item() if probabilities.dim() > 1 else probabilities[idx].item()
            confidences.append(conf)

        # Get the predicted label names
        predicted_labels = [REASON_CATEGORIES[i] for i in predicted_indices]

    return {
        'prediction': predicted_labels,
        'confidence': confidences,
        'probabilities': probabilities.cpu().tolist()
    }

def predict_single_reason(query: str) -> dict:
    """Convenience function to predict a single reason query."""
    try:
        embedding_model, classifier_head = get_reason_models()
        result = predict_reason_query([query], embedding_model, classifier_head)
        
        # Extract values safely
        prediction = result['prediction'][0] if isinstance(result['prediction'], list) else str(result['prediction'])
        confidence = result['confidence'] if isinstance(result['confidence'], float) else (result['confidence'][0] if isinstance(result['confidence'], list) else float(result['confidence']))
        
        # Handle probabilities - ensure it's a list
        probabilities = result['probabilities']
        if isinstance(probabilities, list) and len(probabilities) > 0:
            if isinstance(probabilities[0], list):
                probabilities = probabilities[0]
        
        # Create probability dictionary
        prob_dict = {}
        for i, category in REASON_CATEGORIES.items():
            if i < len(probabilities):
                prob_dict[category] = float(probabilities[i])
            else:
                prob_dict[category] = 0.0
        
        return {
            'query': query,
            'category': prediction,
            'confidence': confidence,
            'probabilities': prob_dict
        }
    except Exception as e:
        # Return a default classification if the model fails
        return {
            'query': query,
            'category': 'GENERAL_MEDICAL',
            'confidence': 0.5,
            'probabilities': {category: 1.0/len(REASON_CATEGORIES) for category in REASON_CATEGORIES.values()},
            'error': str(e)
        }

def test_reason_classifier():
    """Test the reason classifier with sample queries."""
    latest = None
    path = ""
    
    # Try to load the most recent checkpoint
    if os.path.exists(REASON_CHECKPOINT_PATH):
        for d in os.listdir(REASON_CHECKPOINT_PATH):
            if d.endswith('.pt'):
                checkpoint_path = f"{REASON_CHECKPOINT_PATH}/{d}"
                print(f"Found checkpoint: {checkpoint_path}")
                path = checkpoint_path
                break
        
        if not path:
            print("No trained checkpoints found. Using untrained model.")
    else:
        print("No checkpoint directory found. Using untrained model.")

    embedding_model, classifier = get_reason_models()
    
    # Load trained weights if available
    if path and os.path.exists(path):
        try:
            state_dict = torch.load(path, weights_only=True, map_location=DEVICE)
            classifier.load_state_dict(state_dict)
            print(f"Loaded trained weights from {path}")
        except Exception as e:
            print(f"Could not load weights: {e}. Using untrained model.")

    # Test queries for reason classification
    queries = [
        "I have heel pain when I walk",
        "My toenail is ingrown and painful", 
        "I need routine foot care",
        "I sprained my ankle playing sports",
        "I have plantar fasciitis",
        "I need a cortisone injection"
    ]

    print("\nTesting reason classification:")
    pred = predict_reason_query(
        text=queries,
        embedding_model=embedding_model,
        classifier_head=classifier,
    )

    pprint.pprint(pred, indent=4)

if __name__ == "__main__":
    test_reason_classifier()