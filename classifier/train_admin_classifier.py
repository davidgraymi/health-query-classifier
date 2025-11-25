"""
Training Script for Administrative Query Classifier

This script trains the administrative query classifier using synthetic data
and provides evaluation metrics. It addresses the challenge of limited
labeled administrative healthcare query datasets.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifier.admin_classifier import AdminQueryClassifier, ADMIN_CATEGORIES, CATEGORY_DESCRIPTIONS

def create_enhanced_dataset(samples_per_class: int = 100, include_variations: bool = True):
    """
    Create an enhanced synthetic dataset with more realistic variations.
    
    This addresses the challenge mentioned in the bluffing plan about finding
    labeled examples of administrative healthcare queries.
    """
    
    classifier = AdminQueryClassifier()
    base_dataset = classifier.create_synthetic_dataset(samples_per_class)
    
    if not include_variations:
        return base_dataset
    
    # Add contextual variations to make data more realistic
    enhanced_data = []
    
    for _, row in base_dataset.iterrows():
        original_text = row['text']
        label = row['label']
        category = row['category']
        
        # Add original
        enhanced_data.append(row.to_dict())
        
        # Create variations with different phrasings
        if category != 'MEDICAL':  # Keep medical queries as-is for accuracy
            variations = [
                f"Hi, {original_text.lower()}",
                f"Hello, I need help with something. {original_text}",
                f"Can someone assist me? {original_text}",
                f"I have a question about {original_text.lower()}",
                f"Quick question: {original_text.lower()}",
            ]
            
            # Add 1-2 variations per original query
            for variation in variations[:2]:
                enhanced_data.append({
                    'text': variation,
                    'label': label,
                    'category': category
                })
    
    enhanced_df = pd.DataFrame(enhanced_data)
    return enhanced_df.sample(frac=1).reset_index(drop=True)  # Shuffle

def train_classifier(dataset_size: int = 100, epochs: int = 20, output_dir: str = "classifier/admin_model"):
    """Train the administrative classifier with comprehensive evaluation."""
    
    print("=" * 60)
    print("ADMINISTRATIVE QUERY CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Initialize classifier
    print("Initializing classifier...")
    classifier = AdminQueryClassifier()
    
    # Create dataset
    print(f"Creating enhanced dataset with {dataset_size} samples per class...")
    dataset = create_enhanced_dataset(samples_per_class=dataset_size, include_variations=True)
    
    print(f"Total dataset size: {len(dataset)} samples")
    print("\nCategory distribution:")
    category_counts = dataset['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} samples")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_data, test_data = train_test_split(
        dataset, 
        test_size=0.2, 
        stratify=dataset['label'],
        random_state=42
    )
    
    train_data, val_data = train_test_split(
        train_data,
        test_size=0.2,
        stratify=train_data['label'], 
        random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples") 
    print(f"  Test: {len(test_data)} samples")
    
    # Train model
    print(f"\nTraining classifier for {epochs} epochs...")
    metrics = classifier.train(
        train_data=train_data,
        eval_data=val_data,
        epochs=epochs,
        output_dir=output_dir
    )
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    classifier.save_model(output_dir)
    
    # Comprehensive evaluation
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Test set evaluation
    test_results = classifier.evaluate_on_test_set(test_data)
    
    print(f"\nTest Set Accuracy: {test_results['accuracy']:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    report = test_results['classification_report']
    
    for category in ADMIN_CATEGORIES.values():
        if category in report:
            metrics = report[category]
            print(f"\n{category}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  Support: {metrics['support']}")
    
    # Test on example queries
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    example_queries = [
        "I need to schedule an appointment for my annual checkup",
        "My insurance claim was denied and I need help",
        "I have severe chest pain and difficulty breathing", 
        "Can I get copies of my medical records?",
        "What are your office hours on weekends?",
        "I'm experiencing persistent headaches and dizziness",
        "How much will this procedure cost with my insurance?",
        "I need to cancel my appointment tomorrow",
        "My child has had a high fever for 2 days",
        "I need prior authorization for my medication"
    ]
    
    predictions = classifier.predict(example_queries)
    
    for pred in predictions:
        print(f"\nQuery: {pred['query']}")
        print(f"Category: {pred['category']} (confidence: {pred['confidence']:.3f})")
        
        # Show top 2 probabilities
        sorted_probs = sorted(pred['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        print(f"Top predictions: {sorted_probs[0][0]} ({sorted_probs[0][1]:.3f}), "
              f"{sorted_probs[1][0]} ({sorted_probs[1][1]:.3f})")
    
    # Save evaluation results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'test_accuracy': test_results['accuracy'],
            'classification_report': test_results['classification_report'],
            'confusion_matrix': test_results['confusion_matrix'],
            'example_predictions': predictions,
            'training_config': {
                'dataset_size_per_class': dataset_size,
                'epochs': epochs,
                'total_samples': len(dataset)
            }
        }, f, indent=2)
    
    print(f"\nEvaluation results saved to {results_file}")
    
    return classifier, test_results

def create_confusion_matrix_plot(confusion_matrix, categories, output_path):
    """Create and save confusion matrix visualization."""
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='d',
                xticklabels=categories.values(),
                yticklabels=categories.values(),
                cmap='Blues')
    
    plt.title('Administrative Query Classification - Confusion Matrix')
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Administrative Query Classifier')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples per class (default: 100)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--output', type=str, default='classifier/admin_model',
                       help='Output directory for model (default: classifier/admin_model)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate confusion matrix plot')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Train classifier
    classifier, results = train_classifier(
        dataset_size=args.samples,
        epochs=args.epochs,
        output_dir=args.output
    )
    
    # Generate plots if requested
    if args.plot:
        try:
            plot_path = os.path.join(args.output, 'confusion_matrix.png')
            create_confusion_matrix_plot(
                results['confusion_matrix'],
                ADMIN_CATEGORIES,
                plot_path
            )
        except Exception as e:
            print(f"Could not generate plot: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model saved to: {args.output}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    
    # Integration suggestions
    print("\nNext Steps for Integration:")
    print("1. Test the query router with: python classifier/query_router.py")
    print("2. Integrate with existing severity classifier")
    print("3. Deploy in healthcare portal for A/B testing")
    print("4. Collect real-world queries to improve training data")

if __name__ == "__main__":
    main()