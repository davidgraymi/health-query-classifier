"""
Training script for Healthcare Reason Classification

This script trains a classifier for healthcare visit reasons using real
healthcare data. It creates a separate system from the medical/insurance
classifier.
"""

from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, Trainer, TrainingArguments
from head import ClassifierHead
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from pathlib import Path
from datetime import datetime

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
HEALTHCARE_DATA_PATH = "data/reason_for_visit_data.xlsx"
MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"

def get_device():
    """Get the best available device for training/inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_reason_model(num_classes: int):
    """Get model for reason classification."""
    try:
        model_body = SentenceTransformer(
            MODEL_NAME,
            prompts={
                'classification': 'task: healthcare reason classification | query: ',
                'retrieval (query)': 'task: search result | query: ',
                'retrieval (document)': 'title: {title | "none"} | text: ',
            },
            default_prompt_name='classification',
        )
        # Freeze weights of embedding model
        model_head = ClassifierHead(num_classes)
        model = SetFitModel(model_body, model_head)
        model.freeze("body")

    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        raise RuntimeError("Failed to load the embedding model.")
    
    device = get_device()
    print(f"Using device: {device}")
    return model.to(device)

def get_reason_dataset() -> pd.DataFrame:
    """Load the healthcare reason dataset from Excel file."""
    try:
        if not os.path.exists(HEALTHCARE_DATA_PATH):
            raise FileNotFoundError(f"Healthcare data file not found: {HEALTHCARE_DATA_PATH}")
        
        print(f"Loading healthcare data from {HEALTHCARE_DATA_PATH}...")
        df = pd.read_excel(HEALTHCARE_DATA_PATH)
        print(f"Loaded {len(df)} healthcare records")
        return df

    except Exception as e:
        print(f"Error loading healthcare dataset: {e}")
        raise Exception(f"Failed to load healthcare data: {e}")

def map_reason_to_category(reason: str) -> int:
    """Map healthcare reasons to categories using keyword matching."""
    reason_lower = reason.lower()
    
    # ROUTINE_CARE (routine care, maintenance visits)
    if any(word in reason_lower for word in ['routine', 'nail care', 'calluses', 'maintenance']):
        return 0
    
    # PAIN_CONDITIONS (various pain-related conditions)
    elif any(word in reason_lower for word in ['pain', 'ache', 'sore', 'hurt']):
        return 1
    
    # INJURIES (sprains, wounds, trauma)
    elif any(word in reason_lower for word in ['sprain', 'wound', 'injury', 'trauma', 'cut', 'bruise']):
        return 2
    
    # SKIN_CONDITIONS (skin-related issues)
    elif any(word in reason_lower for word in ['ingrown', 'toenail', 'callus', 'corn', 'skin']):
        return 3
    
    # STRUCTURAL_ISSUES (structural problems)
    elif any(word in reason_lower for word in ['flat feet', 'plantar', 'fasciitis', 'achilles', 'tendon', 'arch']):
        return 4
    
    # PROCEDURES (injections, surgical consultations)
    elif any(word in reason_lower for word in ['injection', 'surgical', 'consult', 'postop', 'surgery', 'procedure']):
        return 5
    
    # Default to pain conditions (most common category)
    else:
        return 1

def preprocess_reason_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the healthcare reason dataset for training."""
    training_data = []
    
    for _, row in df.iterrows():
        reason = row['Reason For Visit']
        appointment_type = row.get('Appointment Type', '')
        
        # Map reason to category using keyword matching
        category_id = map_reason_to_category(reason)
        
        # Create enhanced text with context
        enhanced_text = reason
        if pd.notna(appointment_type) and appointment_type:
            enhanced_text += f" | {appointment_type}"
        
        training_data.append({
            'text': enhanced_text,
            'label': category_id,
            'category': REASON_CATEGORIES[category_id],
            'original_reason': reason
        })
    
    processed_df = pd.DataFrame(training_data)
    
    # Show category distribution
    print("\nReason category distribution in training data:")
    for cat_id, cat_name in REASON_CATEGORIES.items():
        count = len(processed_df[processed_df['label'] == cat_id])
        percentage = (count / len(processed_df)) * 100
        print(f"  {cat_name}: {count} samples ({percentage:.1f}%)")
    
    return processed_df

def main():
    print("Healthcare Reason Classification - Training Pipeline")
    print("=" * 60)
    
    # Load and preprocess data
    df = get_reason_dataset()
    df = preprocess_reason_data(df)
    
    # Get model
    model = get_reason_model(len(REASON_CATEGORIES))
    
    # Split data
    train, test = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(train)} samples")
    print(f"  Testing: {len(test)} samples")
    
    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)
    
    # Ensure checkpoint directory exists
    Path(REASON_CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{REASON_CHECKPOINT_PATH}/training_{timestamp}"
    
    args = TrainingArguments(
        output_dir=output_dir,
        # Skip contrastive fine-tuning (body is frozen)
        num_epochs=(0, 20),
        eval_strategy='epoch',
        eval_steps=100,
        save_strategy='epoch',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric='accuracy',
        column_mapping={"text": "text", "label": "label"},
        args=args,
    )

    print("\nStarting reason classification training...")
    trainer.train()

    # Evaluate
    print("\nEvaluating reason classification model...")
    metrics = trainer.evaluate(test_dataset)
    print(f"Final evaluation metrics: {metrics}")
    
    # Save the trained classifier head
    model_save_path = f"{REASON_CHECKPOINT_PATH}/reason_classifier_head_{timestamp}.pt"
    torch.save(model.model_head.state_dict(), model_save_path)
    print(f"Reason classifier head saved to: {model_save_path}")
    
    return metrics

if __name__ == "__main__":
    main()