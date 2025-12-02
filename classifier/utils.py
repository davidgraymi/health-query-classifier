from classifier.head import ClassifierHead

import os
from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"
CLASSIFIER_NAME = "davidgray/health-query-triage"
CATEGORIES: list[str] = ["medical", "insurance"]
CHECKPOINT_PATH = "classifier/checkpoints"
DATETIME_FORMAT = '%Y%m%d_%H%M%S'
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {DEVICE} device")

def get_models(model_id: str | None = None, num_labels: int = len(CATEGORIES)) -> tuple[SentenceTransformer, ClassifierHead]:
    """
    Loads embeddinggemma-300m-medical model and initializes the classification head.
    """
    try:
        model_body = SentenceTransformer(
            MODEL_NAME,
            # prompts={
            #     'retrieval (query)': 'task: search result | query: ',
            #     'retrieval (document)': 'title: {title | "none"} | text: ',
            #     'qa': 'task: question answering | query: ',
            #     'fact verification': 'task: fact checking | query: ',
            #     'classification': 'task: classification | query: ',
            #     'clustering': 'task: clustering | query: ',
            #     'semantic similarity': 'task: sentence similarity | query: ',
            #     'code retrieval': 'task: code retrieval | query: '
            # },
            # default_prompt_name='classification',
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
