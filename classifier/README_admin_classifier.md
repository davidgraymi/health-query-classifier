# Administrative Query Classification System

This module implements a multi-class classifier to differentiate administrative healthcare queries from medical queries, complementing the existing medical severity classification system.

## Overview

The administrative query classifier addresses the challenge of routing healthcare portal queries to appropriate departments. It distinguishes between:

- **Administrative queries** (billing, scheduling, insurance, records, general admin)
- **Medical queries** (symptoms, health concerns - routed to severity classifier)

## Architecture

### Classification Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `BILLING` | Insurance claims, payment issues, billing inquiries | "What is my copay?", "I received a bill but insurance should cover this" |
| `SCHEDULING` | Appointments, cancellations, rescheduling | "I need to schedule an appointment", "Can I reschedule for next week?" |
| `INSURANCE` | Coverage verification, prior authorization, benefits | "Is this procedure covered?", "I need prior authorization" |
| `RECORDS` | Medical records requests, test results, referrals | "I need copies of my lab results", "Can you send my records to another doctor?" |
| `GENERAL_ADMIN` | General administrative questions, contact info | "What are your office hours?", "Where is your clinic located?" |
| `MEDICAL` | Medical symptoms, health concerns | "I have chest pain", "My child has a fever" |

### Technical Implementation

- **Base Model**: `sentence-transformers/embeddinggemma-300m-medical`
- **Architecture**: SetFit with frozen embeddings + trainable classification head
- **Training**: Synthetic dataset with 100+ examples per category
- **Integration**: Works with existing retrieval and routing systems

## Quick Start

### 1. Train the Classifier

```bash
# Train with default settings (100 samples per class, 20 epochs)
python classifier/train_admin_classifier.py

# Custom training
python classifier/train_admin_classifier.py --samples 200 --epochs 30 --output my_model
```

### 2. Use the CLI

```bash
# Classify a single query
python cli/admin_classifier_cli.py "I need to schedule an appointment"

# Use full routing system
python cli/admin_classifier_cli.py --route "I have chest pain"

# Interactive mode
python cli/admin_classifier_cli.py --interactive

# Batch processing
python cli/admin_classifier_cli.py --batch queries.txt --output results.json
```

### 3. Programmatic Usage

```python
from classifier.admin_classifier import AdminQueryClassifier
from classifier.query_router import HealthcareQueryRouter

# Basic classification
classifier = AdminQueryClassifier()
predictions = classifier.predict(["I need to schedule an appointment"])
print(predictions[0]['category'])  # Output: SCHEDULING

# Full routing system
router = HealthcareQueryRouter()
result = router.route_query("I have severe chest pain")
print(result['routing_decision']['department'])  # Output: Emergency Department
```

## System Integration

### Routing Workflow

```
User Query
    ↓
Administrative Classifier
    ↓
┌─────────────────┬─────────────────┐
│  Administrative │     Medical     │
│     Queries     │     Queries     │
│        ↓        │        ↓        │
│   Department    │   Severity      │
│    Routing      │ Classification  │
│        ↓        │        ↓        │
│   • Billing     │   • Emergency   │
│   • Scheduling  │   • Urgent      │
│   • Insurance   │   • Routine     │
│   • Records     │                 │
│   • General     │                 │
└─────────────────┴─────────────────┘
```

### Integration with Existing Systems

The administrative classifier integrates seamlessly with:

1. **Retrieval System** (`retriever/`): Provides relevant documents for medical queries
2. **Severity Classification**: Routes medical queries through severity assessment
3. **Portal Interface**: Enables intelligent query routing in healthcare portals

## Training Data Strategy

### Synthetic Data Generation

Due to the challenge of finding labeled administrative healthcare queries, the system uses synthetic data generation:

```python
# Categories with realistic examples
synthetic_data = {
    "BILLING": [
        "What is my copay for this visit?",
        "I received a bill but my insurance should cover this",
        # ... 100+ examples per category
    ],
    # ... other categories
}
```

### Data Enhancement

- **Contextual Variations**: Adds natural language variations
- **Balanced Dataset**: Equal representation across categories
- **Realistic Phrasing**: Healthcare-specific terminology and scenarios

## Performance Metrics

### Expected Performance
- **Accuracy**: >90% on synthetic test data
- **Precision/Recall**: Balanced across all categories
- **Confidence**: High confidence (>0.8) for clear administrative queries

### Evaluation Framework

```bash
# Generate comprehensive evaluation
python classifier/train_admin_classifier.py --samples 150 --plot

# Results include:
# - Classification report
# - Confusion matrix
# - Example predictions
# - Confidence analysis
```

## Deployment Considerations

### Production Readiness

1. **Model Persistence**: Trained models saved with category mappings
2. **Error Handling**: Graceful fallbacks for edge cases
3. **Confidence Thresholds**: Configurable confidence levels for routing decisions
4. **Monitoring**: Built-in evaluation and statistics tracking

### Scalability

- **Batch Processing**: Efficient handling of multiple queries
- **Caching**: Integration with existing cache system
- **Device Support**: CPU/GPU/MPS compatibility

## Future Enhancements

### Real-World Data Collection

1. **Portal Integration**: Collect actual healthcare portal queries
2. **Active Learning**: Improve model with user feedback
3. **Domain Adaptation**: Fine-tune for specific healthcare systems

### Advanced Features

1. **Multi-Intent Detection**: Handle queries with multiple intents
2. **Confidence Calibration**: Improve confidence score reliability
3. **Hierarchical Classification**: Sub-categories within administrative types

## API Reference

### AdminQueryClassifier

```python
class AdminQueryClassifier:
    def __init__(self, model_path: Optional[str] = None)
    def predict(self, queries: List[str]) -> List[Dict]
    def train(self, train_data: pd.DataFrame, eval_data: Optional[pd.DataFrame] = None)
    def save_model(self, path: str)
    def load_model(self, path: str)
```

### HealthcareQueryRouter

```python
class HealthcareQueryRouter:
    def __init__(self, admin_model_path: Optional[str] = None)
    def route_query(self, query: str) -> Dict
    def batch_route_queries(self, queries: List[str]) -> List[Dict]
    def get_routing_statistics(self, queries: List[str]) -> Dict
```

## Contributing

### Adding New Categories

1. Update `ADMIN_CATEGORIES` in `admin_classifier.py`
2. Add examples to synthetic data generation
3. Update routing rules in `query_router.py`
4. Retrain the model

### Improving Training Data

1. Add real-world examples to synthetic dataset
2. Implement few-shot learning approaches
3. Create domain-specific variations

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure model path exists and contains required files
2. **Low Confidence**: May indicate need for more training data or model retraining
3. **Misclassification**: Check if query fits existing categories or needs new category

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model predictions with probabilities
predictions = classifier.predict(["ambiguous query"])
print(predictions[0]['probabilities'])
```

## License

This module is part of the health-query-classifier project and follows the same licensing terms.