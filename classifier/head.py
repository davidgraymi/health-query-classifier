from typing import Dict
from torch import nn
import torch
from huggingface_hub import PyTorchModelHubMixin

class ClassifierHead(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://huggingface.co/davidgray/health-query-triage",
    pipeline_tag="text-classification",
    library_name="PyTorch",
    tags=["medical", "classification"],
):
    def __init__(self, num_classes: int, embedding_dim: int = 768): # Embedding-Gemma-300M has a 768-dimensional output
        super().__init__()

        self.linear_elu_stack = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates logits from the sentence embedding.
        
        Args:
            features (Dict[str, torch.Tensor]): Output dictionary from the Sentence Transformer body, 
                                                containing 'sentence_embedding'.
        Returns:
            Dict[str, torch.Tensor]: Dictionary with the 'logits' key.
        """
        embeddings = features['sentence_embedding']
        logits = self.linear_elu_stack(embeddings)
        return {"logits": logits}
    
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Classifies embeddings into integer labels in the range [0, num_classes).
        
        Args:
            embeddings (torch.Tensor): Tensor with shape [num_inputs, embedding_size].
        
        Returns:
            torch.Tensor: Integer labels with shape [num_inputs].
        """
        # Get probabilities and find the class with the highest probability
        proba = self.predict_proba(embeddings)
        return torch.argmax(proba, dim=-1)

    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Classifies embeddings into probabilities for each class (summing to 1).
        
        Args:
            embeddings (torch.Tensor): Tensor with shape [num_inputs, embedding_size].
            
        Returns:
            torch.Tensor: Float probabilities with shape [num_inputs, num_classes].
        """
        # Apply the forward pass of the head to get logits
        self.eval()
        with torch.no_grad():
            logits = self.linear_elu_stack(embeddings)
            # Convert logits to probabilities using Softmax
            probabilities = self.softmax(logits)
        self.train() # Set back to training mode
        
        return probabilities

    def get_loss_fn(self) -> nn.Module:
        """
        Returns an initialized loss function for training.
        
        Returns:
            nn.Module: An initialized loss function (e.g., CrossEntropyLoss).
        """
        # CrossEntropyLoss expects logits (raw scores) as input
        return nn.CrossEntropyLoss()
