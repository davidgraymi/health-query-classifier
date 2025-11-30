"""
Healthcare Reason for Visit Classifier

This module implements a classifier for healthcare clinic queries
using real healthcare data from clinic appointment records.

Categories based on the actual data:
- ROUTINE_CARE: Routine care, maintenance visits
- PAIN_CONDITIONS: Various pain-related conditions
- INJURIES: Sprains, wounds, trauma-related visits
- SKIN_CONDITIONS: Skin-related conditions and issues
- STRUCTURAL_ISSUES: Structural problems and conditions
- PROCEDURES: Injections, surgical consults, postop care
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset
import json

from classifier.head import ClassifierHead

# Healthcare reason categories based on real data analysis
REASON_CATEGORIES = {
    0: "ROUTINE_CARE",
    1: "PAIN_CONDITIONS",
    2: "INJURIES",
    3: "SKIN_CONDITIONS",
    4: "STRUCTURAL_ISSUES",
    5: "PROCEDURES"
}

CATEGORY_DESCRIPTIONS = {
    "ROUTINE_CARE": "Routine healthcare, maintenance visits, general care",
    "PAIN_CONDITIONS": "Various pain-related conditions and discomfort",
    "INJURIES": "Sprains, wounds, trauma-related conditions",
    "SKIN_CONDITIONS": "Skin-related issues and conditions",
    "STRUCTURAL_ISSUES": "Structural problems and related conditions",
    "PROCEDURES": "Injections, surgical consultations, post-operative care"
}

class ReasonClassifier:
    """
    Healthcare Reason Classifier that uses real clinic data to classify
    patient queries into specific healthcare reason categories.
    """
    
    def __init__(self, data_file: str = "data/reason_for_visit_data.xlsx"):
        self.model_name = "sentence-transformers/embeddinggemma-300m-medical"
        self.num_classes = len(REASON_CATEGORIES)
        self.categories = REASON_CATEGORIES
        self.data_file = data_file
        self.model = None
        self.device = self._get_device()
        
        # Load and process real data
        self.healthcare_df = self._load_data()
        self._initialize_model()
    
    def _get_device(self):
        """Get the best available device for training/inference."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_data(self) -> pd.DataFrame:
        """Load the real healthcare dataset."""
        try:
            df = pd.read_excel(self.data_file)
            print(f"Loaded {len(df)} healthcare records from {self.data_file}")
            print(f"Unique reasons: {df['Reason For Visit'].nunique()}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise RuntimeError(f"Failed to load healthcare data from {self.data_file}")
    
    def _initialize_model(self):
        """Initialize the model with the existing infrastructure."""
        try:
            model_body = SentenceTransformer(
                self.model_name,
                prompts={
                    'classification': 'task: healthcare reason classification | query: ',
                    'retrieval (query)': 'task: search result | query: ',
                    'retrieval (document)': 'title: {title | "none"} | text: ',
                },
                default_prompt_name='classification',
            )
            
            model_head = ClassifierHead(self.num_classes, embedding_dim=768)
            self.model = SetFitModel(model_body, model_head)
            self.model.freeze("body")  # Freeze embedding weights
            self.model = self.model.to(self.device)
            
            print(f"Initialized ReasonClassifier on {self.device}")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise RuntimeError("Failed to initialize reason classifier")
    
    def _map_reason_to_category(self, reason: str) -> int:
        """
        Map real healthcare reasons to categories using keyword matching.
        Based on the actual data distribution.
        """
        reason_lower = reason.lower()
        
        # ROUTINE_CARE (routine foot care, nail care, calluses)
        if any(word in reason_lower for word in ['routine', 'nail care', 'calluses']):
            return 0
        
        # PAIN_CONDITIONS (heel pain, ankle pain, foot pain, etc.)
        if any(word in reason_lower for word in ['pain', 'ache', 'sore']):
            return 1
        
        # INJURIES (ankle sprain, wounds, trauma)
        if any(word in reason_lower for word in ['sprain', 'wound', 'injury', 'trauma']):
            return 2
        
        # SKIN_CONDITIONS (ingrown toenail, calluses, skin issues)
        if any(word in reason_lower for word in ['ingrown', 'toenail', 'callus', 'skin']):
            return 3
        
        # STRUCTURAL_ISSUES (flat feet, plantar fasciitis, achilles)
        if any(word in reason_lower for word in ['flat feet', 'plantar', 'fasciitis', 'achilles', 'tendon']):
            return 4
        
        # PROCEDURES (injection, surgical consult, postop)
        if any(word in reason_lower for word in ['injection', 'surgical', 'consult', 'postop', 'procedure']):
            return 5
        
        # Default to pain conditions (most common category)
        return 1
    
    def create_real_dataset(self) -> pd.DataFrame:
        """
        Create training dataset from real healthcare data.
        """
        
        training_data = []
        
        for _, row in self.healthcare_df.iterrows():
            reason = row['Reason For Visit']
            appointment_type = row['Appointment Type']
            
            # Map reason to category
            category_id = self._map_reason_to_category(reason)
            
            # Create enhanced text with context
            enhanced_text = reason
            if pd.notna(appointment_type):
                enhanced_text += f" | {appointment_type}"
            
            training_data.append({
                'text': enhanced_text,
                'label': category_id,
                'category': self.categories[category_id],
                'original_reason': reason,
                'appointment_type': appointment_type
            })
        
        df = pd.DataFrame(training_data)
        
        # Show category distribution
        print("\nCategory distribution in training data:")
        for cat_id, cat_name in self.categories.items():
            count = len(df[df['label'] == cat_id])
            percentage = (count / len(df)) * 100
            print(f"  {cat_name}: {count} samples ({percentage:.1f}%)")
        
        return df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    def train(self, train_data: pd.DataFrame = None, eval_data: Optional[pd.DataFrame] = None, 
              epochs: int = 16, output_dir: str = "classifier/reason_checkpoints"):
        """Train the healthcare reason classifier."""
        
        if train_data is None:
            train_data = self.create_real_dataset()
        
        if eval_data is None:
            train_data, eval_data = train_test_split(train_data, test_size=0.2, 
                                                   stratify=train_data['label'], 
                                                   random_state=42)
        
        train_dataset = Dataset.from_pandas(train_data)
        eval_dataset = Dataset.from_pandas(eval_data)
        
        from setfit import Trainer, TrainingArguments
        
        args = TrainingArguments(
            output_dir=output_dir,
            num_epochs=(0, epochs),  # Skip contrastive learning, only train head
            eval_strategy='epoch',
            eval_steps=100,
            save_strategy='epoch',
            logging_steps=50,
        )
        
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metric='accuracy',
            column_mapping={"text": "text", "label": "label"},
            args=args,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Evaluate
        metrics = trainer.evaluate(eval_dataset)
        print(f"Training completed. Final metrics: {metrics}")
        
        return metrics
    
    def predict(self, queries: List[str]) -> List[Dict]:
        """
        Predict healthcare reason categories for a list of queries.
        
        Returns:
            List of dictionaries with 'query', 'category', 'confidence', 'probabilities'
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Train or load a model first.")
        
        predictions = []
        
        for query in queries:
            # Get prediction using SetFit's built-in methods
            pred_label = self.model.predict([query])[0]
            pred_proba = self.model.predict_proba([query])[0]
            
            category = self.categories[int(pred_label)]
            confidence = float(pred_proba[int(pred_label)])
            
            predictions.append({
                'query': query,
                'category': category,
                'confidence': confidence,
                'probabilities': {self.categories[i]: float(prob) 
                               for i, prob in enumerate(pred_proba)}
            })
        
        return predictions
    
    def save_model(self, path: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_pretrained(path)
        
        # Save category mapping
        with open(os.path.join(path, 'categories.json'), 'w') as f:
            json.dump(self.categories, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.model = SetFitModel.from_pretrained(path)
        self.model = self.model.to(self.device)
        
        # Load category mapping
        with open(os.path.join(path, 'categories.json'), 'r') as f:
            self.categories = {int(k): v for k, v in json.load(f).items()}
        
        print(f"Model loaded from {path}")
    
    def evaluate_on_test_set(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate the model on a test dataset."""
        predictions = self.predict(test_data['text'].tolist())
        
        y_true = test_data['label'].tolist()
        y_pred = [list(self.categories.keys())[list(self.categories.values()).index(p['category'])] 
                  for p in predictions]
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=list(self.categories.values()),
                                     output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': report['accuracy']
        }
    
    def analyze_real_data(self):
        """Analyze the real healthcare data to understand patterns."""
        print("Real Data Analysis:")
        print("=" * 50)
        
        print(f"Total records: {len(self.healthcare_df)}")
        print(f"Unique reasons: {self.healthcare_df['Reason For Visit'].nunique()}")
        
        print("\nTop 15 reasons for visit:")
        top_reasons = self.healthcare_df['Reason For Visit'].value_counts().head(15)
        for reason, count in top_reasons.items():
            category_id = self._map_reason_to_category(reason)
            category_name = self.categories[category_id]
            print(f"  {reason}: {count} ({category_name})")
        
        print(f"\nAppointment types:")
        print(self.healthcare_df['Appointment Type'].value_counts())


def main():
    """Example usage and training script for healthcare reason data."""
    print("Initializing Healthcare Reason Classifier...")
    
    # Initialize classifier with real data
    classifier = ReasonClassifier()
    
    # Analyze the real data
    classifier.analyze_real_data()
    
    # Create training dataset from real data
    print("\nCreating training dataset from real healthcare data...")
    dataset = classifier.create_real_dataset()
    
    print(f"Dataset created with {len(dataset)} real examples")
    
    # Train the model
    print("\nTraining classifier...")
    metrics = classifier.train(dataset, epochs=20)
    
    # Save the model
    model_path = "classifier/reason_model"
    classifier.save_model(model_path)
    
    # Test predictions on healthcare reason queries
    test_queries = [
        "I have heel pain when I walk",
        "My toenail is ingrown and painful",
        "I need routine foot care",
        "I sprained my ankle playing sports",
        "I have flat feet and need evaluation",
        "I need a cortisone injection for my foot",
        "I have plantar fasciitis",
        "My foot wound is not healing"
    ]
    
    print("\nTesting predictions on healthcare reason queries:")
    predictions = classifier.predict(test_queries)
    
    for pred in predictions:
        print(f"Query: {pred['query']}")
        print(f"Category: {pred['category']} (confidence: {pred['confidence']:.3f})")
        print("---")


if __name__ == "__main__":
    main()