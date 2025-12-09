import json
from dataclasses import asdict
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from classifier.head import ClassifierHead
from classifier.infer import predict_query
from classifier.utils import get_models
from retriever import Retriever
from team.candidates import get_candidates, _available
from config import settings

class HealthQueryPipeline:
    def __init__(self, use_reranker: bool = False):
        self.use_reranker = use_reranker
        self.embedding_model: Optional[SentenceTransformer] = None
        self.classifier: Optional[ClassifierHead] = None
        self.retriever: Optional[Retriever] = None
        self.is_initialized = False

    def initialize(self):
        """Loads models and initializes the retriever."""
        if self.is_initialized:
            return

        print(f"Loading embedding model: {settings.MODEL_NAME}...")
        self.embedding_model, self.classifier = get_models(model_id=settings.CLASSIFIER_NAME)
        print("Model loaded.")

        print("Initializing retriever...")
        cfg = _available(settings.CORPORA_CONFIG)
        if not cfg:
            raise RuntimeError("No corpora files found in data/corpora. Build them first.")

        self.retriever = Retriever(
            corpora_config=cfg,
            use_reranker=self.use_reranker,
            embedding_model=self.embedding_model
        )
        print("Retriever initialized.")
        self.is_initialized = True

    def predict(self, query: str, k: int = 10) -> Dict[str, Any]:
        """
        Runs the full pipeline: Classification -> Retrieval (if medical).
        """
        if not self.is_initialized:
            self.initialize()

        classification = predict_query(
            text=[query],
            embedding_model=self.embedding_model,
            classifier_head=self.classifier,
        )

        predictions = classification["prediction"]
        result = {
            "query": query,
            "classification": {
                "prediction": predictions[0],
                "probabilities": {
                    cat: prob
                    for cat, prob in zip(settings.CATEGORIES, classification['probabilities'])
                }
            },
            "retrieval": []
        }

        if "medical" in predictions:
            hits = get_candidates(
                query=query,
                retriever=self.retriever,
                k_retrieve=k,
            )
            result["retrieval"] = [asdict(hit) for hit in hits]

        return result

    def get_index_progress(self):
        """Returns (current, total) of the underlying index."""
        if not self.retriever:
            return 0, 0
        return self.retriever.get_index_progress()
