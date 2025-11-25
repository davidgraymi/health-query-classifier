"""
Administrative Query Classification System

This module implements a multi-class classifier to differentiate administrative 
healthcare queries from medical queries, complementing the existing severity 
classification system.

Categories:
- BILLING: Insurance claims, payment issues, billing inquiries
- SCHEDULING: Appointments, cancellations, rescheduling
- INSURANCE: Coverage verification, prior authorization, benefits
- RECORDS: Medical records requests, test results, referrals
- GENERAL_ADMIN: General administrative questions, contact info
- MEDICAL: Medical symptoms, health concerns (routes to severity classifier)
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

# Administrative query categories
ADMIN_CATEGORIES = {
    0: "BILLING",
    1: "SCHEDULING", 
    2: "INSURANCE",
    3: "RECORDS",
    4: "GENERAL_ADMIN",
    5: "MEDICAL"
}

CATEGORY_DESCRIPTIONS = {
    "BILLING": "Insurance claims, payment issues, billing inquiries, copay questions",
    "SCHEDULING": "Appointments, cancellations, rescheduling, availability",
    "INSURANCE": "Coverage verification, prior authorization, benefits, eligibility",
    "RECORDS": "Medical records requests, test results, referrals, documentation",
    "GENERAL_ADMIN": "General administrative questions, contact information, hours",
    "MEDICAL": "Medical symptoms, health concerns, clinical questions"
}

class AdminQueryClassifier:
    """
    Administrative Query Classifier that leverages the existing embedding infrastructure
    to classify healthcare portal queries into administrative vs medical categories.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_name = "sentence-transformers/embeddinggemma-300m-medical"
        self.num_classes = len(ADMIN_CATEGORIES)
        self.categories = ADMIN_CATEGORIES
        self.model = None
        self.device = self._get_device()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_model()
    
    def _get_device(self):
        """Get the best available device for training/inference."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _initialize_model(self):
        """Initialize the model with the existing infrastructure."""
        try:
            model_body = SentenceTransformer(
                self.model_name,
                prompts={
                    'classification': 'task: administrative query classification | query: ',
                    'retrieval (query)': 'task: search result | query: ',
                    'retrieval (document)': 'title: {title | "none"} | text: ',
                },
                default_prompt_name='classification',
            )
            
            model_head = ClassifierHead(self.num_classes, embedding_dim=768)
            self.model = SetFitModel(model_body, model_head)
            self.model.freeze("body")  # Freeze embedding weights
            self.model = self.model.to(self.device)
            
            print(f"Initialized AdminQueryClassifier on {self.device}")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise RuntimeError("Failed to initialize administrative query classifier")
    
    def create_synthetic_dataset(self, samples_per_class: int = 50) -> pd.DataFrame:
        """
        Create synthetic training data for administrative query classification.
        This addresses the challenge of finding labeled administrative healthcare queries.
        """
        
        # Synthetic examples for each category
        synthetic_data = {
            "BILLING": [
                "What is my copay for this visit?",
                "I received a bill but my insurance should cover this",
                "How much will this procedure cost?",
                "My insurance claim was denied, what should I do?",
                "Can I set up a payment plan for my medical bills?",
                "I need to update my insurance information",
                "Why was I charged for this service?",
                "What does this charge on my bill mean?",
                "I need a receipt for my payment",
                "How do I submit a claim to my insurance?",
                "My deductible amount seems wrong",
                "Can I get an estimate for this treatment?",
                "I was double charged for my visit",
                "How do I dispute a billing error?",
                "What payment methods do you accept?",
                "I need help understanding my medical bill",
                "Can I get financial assistance for treatment?",
                "My insurance card has new information",
                "I need to pay my outstanding balance",
                "What is the cost of lab work?"
            ],
            
            "SCHEDULING": [
                "I need to schedule an appointment",
                "Can I reschedule my appointment for next week?",
                "What times are available for Dr. Smith?",
                "I need to cancel my appointment tomorrow",
                "How far in advance can I book an appointment?",
                "I need an urgent appointment today",
                "Can I schedule a follow-up visit?",
                "What is the earliest available appointment?",
                "I need to change my appointment time",
                "Can I book a telehealth appointment?",
                "I'm running late for my appointment",
                "How do I schedule a specialist referral?",
                "Can I see a different doctor?",
                "I need to schedule my annual checkup",
                "What is your cancellation policy?",
                "Can I schedule multiple appointments at once?",
                "I need an appointment for my child",
                "How long is the wait time for new patients?",
                "Can I schedule an appointment online?",
                "I need to confirm my appointment time"
            ],
            
            "INSURANCE": [
                "Is this procedure covered by my insurance?",
                "I need prior authorization for treatment",
                "Can you verify my insurance benefits?",
                "What is my insurance coverage for prescriptions?",
                "I need to add my spouse to my insurance",
                "My insurance requires a referral",
                "What specialists are in my network?",
                "I need to update my insurance card information",
                "Does my plan cover preventive care?",
                "What is my out-of-network coverage?",
                "I need help with insurance pre-approval",
                "Can you check if my insurance is active?",
                "What is my copay for specialist visits?",
                "I need a letter for my insurance company",
                "Does my insurance cover this medication?",
                "I need to file an insurance appeal",
                "What is my deductible amount?",
                "Can you submit this to my insurance?",
                "I need proof of coverage for my employer",
                "My insurance information has changed"
            ],
            
            "RECORDS": [
                "I need copies of my medical records",
                "Can you send my test results to another doctor?",
                "I need a referral to a specialist",
                "Can I get my lab results online?",
                "I need my vaccination records",
                "Can you fax my records to my new doctor?",
                "I need a copy of my prescription history",
                "Can I access my medical records online?",
                "I need documentation for work",
                "Can you send my X-ray results?",
                "I need my medical records for insurance",
                "Can I get a summary of my visit?",
                "I need my test results explained",
                "Can you release my records to my family?",
                "I need a medical clearance letter",
                "Can I get my immunization history?",
                "I need my records transferred to another clinic",
                "Can you provide my medication list?",
                "I need documentation for disability",
                "Can I get a copy of my treatment plan?"
            ],
            
            "GENERAL_ADMIN": [
                "What are your office hours?",
                "Where is your clinic located?",
                "How do I contact the nurse?",
                "What is your phone number?",
                "Do you have parking available?",
                "What should I bring to my appointment?",
                "How do I reach the on-call doctor?",
                "What is your address?",
                "Do you have wheelchair access?",
                "How do I contact billing department?",
                "What forms do I need to fill out?",
                "Can I speak to the office manager?",
                "What is your fax number?",
                "How do I leave a message for my doctor?",
                "Do you offer patient portal access?",
                "What languages do you speak?",
                "How do I update my contact information?",
                "What is your email address?",
                "Do you have Saturday hours?",
                "How do I file a complaint?"
            ],
            
            "MEDICAL": [
                "I have chest pain and shortness of breath",
                "My child has a fever and rash",
                "I'm experiencing severe headaches",
                "I have been coughing for two weeks",
                "My blood pressure seems high",
                "I have stomach pain after eating",
                "I'm feeling dizzy and nauseous",
                "I have a painful lump in my neck",
                "My ankle is swollen and painful",
                "I'm having trouble sleeping",
                "I have been losing weight unexpectedly",
                "My vision has been blurry lately",
                "I have persistent back pain",
                "I'm experiencing anxiety symptoms",
                "I have a suspicious mole that changed",
                "My joints are stiff and painful",
                "I have been having heart palpitations",
                "I'm concerned about my diabetes",
                "I have recurring urinary tract infections",
                "I'm experiencing memory problems"
            ]
        }
        
        # Create balanced dataset
        data_rows = []
        for category, examples in synthetic_data.items():
            label = [k for k, v in ADMIN_CATEGORIES.items() if v == category][0]
            
            # Use all provided examples and generate variations if needed
            selected_examples = examples[:samples_per_class] if len(examples) >= samples_per_class else examples
            
            # If we need more examples, create variations
            while len(selected_examples) < samples_per_class:
                # Simple variation by adding context
                base_example = np.random.choice(examples)
                variations = [
                    f"Hi, {base_example.lower()}",
                    f"Hello, {base_example.lower()}",
                    f"Can you help me? {base_example}",
                    f"I have a question: {base_example.lower()}",
                    f"Quick question - {base_example.lower()}"
                ]
                selected_examples.extend(variations[:samples_per_class - len(selected_examples)])
            
            for example in selected_examples[:samples_per_class]:
                data_rows.append({
                    'text': example,
                    'label': label,
                    'category': category
                })
        
        df = pd.DataFrame(data_rows)
        return df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    def train(self, train_data: pd.DataFrame, eval_data: Optional[pd.DataFrame] = None, 
              epochs: int = 16, output_dir: str = "classifier/admin_checkpoints"):
        """Train the administrative query classifier."""
        
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
        Predict administrative categories for a list of queries.
        
        Returns:
            List of dictionaries with 'query', 'category', 'confidence', 'probabilities'
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Train or load a model first.")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for query in queries:
                # Get prediction
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


def main():
    """Example usage and training script."""
    print("Initializing Administrative Query Classifier...")
    
    # Initialize classifier
    classifier = AdminQueryClassifier()
    
    # Create synthetic dataset
    print("Creating synthetic training dataset...")
    dataset = classifier.create_synthetic_dataset(samples_per_class=100)
    
    print(f"Dataset created with {len(dataset)} samples")
    print("Category distribution:")
    print(dataset['category'].value_counts())
    
    # Train the model
    print("\nTraining classifier...")
    metrics = classifier.train(dataset, epochs=20)
    
    # Save the model
    model_path = "classifier/admin_model"
    classifier.save_model(model_path)
    
    # Test predictions
    test_queries = [
        "I need to schedule an appointment for next week",
        "My insurance claim was denied",
        "I have severe chest pain",
        "Can I get copies of my medical records?",
        "What are your office hours?",
        "I'm experiencing shortness of breath and dizziness"
    ]
    
    print("\nTesting predictions:")
    predictions = classifier.predict(test_queries)
    
    for pred in predictions:
        print(f"Query: {pred['query']}")
        print(f"Category: {pred['category']} (confidence: {pred['confidence']:.3f})")
        print("---")


if __name__ == "__main__":
    main()