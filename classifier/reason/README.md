# Healthcare Reason Classification System

This module implements a specialized classifier for healthcare visit reasons using real clinic data to classify patient queries into specific healthcare reason categories.

## Overview

The reason classifier addresses the challenge of routing medical healthcare queries to appropriate specialized departments. It classifies medical queries into specific reason categories based on actual healthcare visit data.

## Architecture

### Classification Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `ROUTINE_CARE` | Routine healthcare, maintenance visits, general care | "I need routine foot care", "Regular nail care appointment" |
| `PAIN_CONDITIONS` | Various pain-related conditions and discomfort | "I have heel pain when I walk", "My ankle is sore" |
| `INJURIES` | Sprains, wounds, trauma-related conditions | "I sprained my ankle playing sports", "I have a wound that won't heal" |
| `SKIN_CONDITIONS` | Skin-related issues and conditions | "My toenail is ingrown and infected", "I have calluses on my feet" |
| `STRUCTURAL_ISSUES` | Structural problems and related conditions | "I have flat feet", "I need evaluation for plantar fasciitis" |
| `PROCEDURES` | Injections, surgical consultations, post-operative care | "I need a cortisone injection", "Post-surgical follow-up" |

### Technical Implementation

- **Base Model**: `sentence-transformers/embeddinggemma-300m-medical`
- **Architecture**: SetFit with frozen embeddings + trainable classification head
- **Training**: Real healthcare data from clinic appointment records
- **Integration**: Works as part of the complete healthcare routing system

## Quick Start

### 1. Train the Classifier

```bash
# Train with real healthcare data
python classifier/reason/train_reason.py

# The training script will:
# - Load real healthcare data from data/reason_for_visit_data.xlsx
# - Map reasons to categories using keyword matching
# - Train the classifier with frozen embeddings
# - Save the trained model to classifier/reason_checkpoints/
```

### 2. Use the CLI

```bash
# Classify a single reason query
python cli/reason_classifier_cli_new.py "I have heel pain when I walk"

# Interactive mode
python cli/reason_classifier_cli_new.py --interactive

# Batch processing
python cli/reason_classifier_cli_new.py --batch queries.txt --output results.json

# Use complete healthcare routing system
python cli/healthcare_classifier_cli.py "I need routine foot care"
```

### 3. Programmatic Usage

```python
from classifier.reason import ReasonClassifier, predict_single_reason

# Using the main classifier class
classifier = ReasonClassifier()
predictions = classifier.predict(["I have heel pain when I walk"])
print(predictions[0]['category'])  # Output: PAIN_CONDITIONS

# Using convenience function
result = predict_single_reason("I need routine foot care")
print(result['category'])  # Output: ROUTINE_CARE
print(result['confidence'])  # Confidence score
print(result['probabilities'])  # All category probabilities
```

## System Integration

### Complete Healthcare Routing Workflow

```
User Query
    ↓
Medical vs Insurance Classification
    ↓
┌─────────────────┬─────────────────┐
│   Insurance     │     Medical     │
│   Queries       │     Queries     │
│       ↓         │        ↓        │
│  Insurance      │   Reason        │
│  Department     │ Classification  │
│                 │        ↓        │
│                 │  • ROUTINE_CARE │
│                 │  • PAIN_CONDITIONS │
│                 │  • INJURIES     │
│                 │  • SKIN_CONDITIONS │
│                 │  • STRUCTURAL_ISSUES │
│                 │  • PROCEDURES   │
└─────────────────┴─────────────────┘
```

### Integration with Healthcare System

The reason classifier integrates as part of the complete healthcare routing system:

1. **Primary Classification**: Medical vs Insurance queries
2. **Reason Classification**: Medical queries → Specific reason categories
3. **Department Routing**: Route to appropriate specialized departments

## Training Data Strategy

### Real Healthcare Data

The system uses actual healthcare clinic data:

```python
# Data source: data/reason_for_visit_data.xlsx
# Contains real patient visit reasons and appointment types
# Examples from actual data:
# - "Heel pain"
# - "Routine foot care"
# - "Ingrown toenail"
# - "Ankle sprain"
# - "Plantar fasciitis"
```

### Category Mapping Strategy

The system uses keyword-based mapping to categorize real healthcare reasons:

```python
def map_reason_to_category(reason: str) -> int:
    reason_lower = reason.lower()
    
    # ROUTINE_CARE (routine care, maintenance visits)
    if any(word in reason_lower for word in ['routine', 'nail care', 'calluses']):
        return 0
    
    # PAIN_CONDITIONS (various pain-related conditions)
    elif any(word in reason_lower for word in ['pain', 'ache', 'sore']):
        return 1
    
    # ... other categories
```

## Performance Metrics

### Expected Performance
- **Accuracy**: Based on real healthcare data patterns
- **Categories**: 6 specialized healthcare reason categories
- **Confidence**: Variable based on training data quality

### Evaluation Framework

```bash
# Train and evaluate the model
python classifier/reason/train_reason.py

# Test the trained model
python classifier/reason/infer_reason.py

# Results include:
# - Training metrics
# - Category distribution
# - Example predictions with confidence scores
```

## File Structure

```
classifier/reason/
├── __init__.py              # Package initialization and exports
├── README.md               # This documentation
├── reason_classifier.py    # Main ReasonClassifier class
├── infer_reason.py        # Inference functions and utilities
└── train_reason.py        # Training script and functions
```

## API Reference

### ReasonClassifier

```python
class ReasonClassifier:
    def __init__(self, data_file: str = "data/reason_for_visit_data.xlsx")
    def predict(self, queries: List[str]) -> List[Dict]
    def train(self, train_data: pd.DataFrame = None, eval_data: Optional[pd.DataFrame] = None)
    def save_model(self, path: str)
    def load_model(self, path: str)
    def create_real_dataset(self) -> pd.DataFrame
    def analyze_real_data(self)
```

### Inference Functions

```python
def predict_single_reason(query: str) -> dict
def predict_reason_query(text: list[str], embedding_model, classifier_head) -> dict
def get_reason_models() -> tuple
def test_reason_classifier()
```

### Training Functions

```python
def get_reason_model(num_classes: int)
def get_reason_dataset() -> pd.DataFrame
def map_reason_to_category(reason: str) -> int
def preprocess_reason_data(df: pd.DataFrame) -> pd.DataFrame
```

## Data Requirements

### Healthcare Data Format

The system expects healthcare data in Excel format with these columns:

```
Required columns:
- "Reason For Visit": The primary reason for the healthcare visit
- "Appointment Type": Type of appointment (optional, used for context)

Example data:
| Reason For Visit | Appointment Type |
|------------------|------------------|
| Heel pain        | Follow-up        |
| Routine foot care| Maintenance      |
| Ingrown toenail  | New Patient      |
```

## Deployment Considerations

### Production Readiness

1. **Model Persistence**: Trained models saved with timestamps in `classifier/reason_checkpoints/`
2. **Error Handling**: Graceful fallbacks for prediction failures
3. **Real Data Integration**: Uses actual healthcare clinic data
4. **Device Support**: CPU/GPU/MPS compatibility

### Scalability

- **Batch Processing**: Efficient handling of multiple queries
- **Integration**: Works with existing healthcare routing system
- **Checkpoints**: Automatic model saving with timestamps

## Future Enhancements

### Data Improvements

1. **Expanded Dataset**: Include more healthcare specialties
2. **Active Learning**: Improve model with real-world feedback
3. **Multi-language Support**: Support for non-English healthcare queries

### Advanced Features

1. **Confidence Calibration**: Improve confidence score reliability
2. **Hierarchical Classification**: Sub-categories within reason types
3. **Context Awareness**: Consider patient history and appointment context

## Troubleshooting

### Common Issues

1. **Data Loading Errors**: Ensure `data/reason_for_visit_data.xlsx` exists
2. **Low Confidence**: May indicate need for more training data or model retraining
3. **Import Errors**: Ensure all dependencies are installed and paths are correct

### Debug Mode

```python
# Test the classifier with sample queries
from classifier.reason.infer_reason import test_reason_classifier
test_reason_classifier()

# Check model predictions with probabilities
from classifier.reason import predict_single_reason
result = predict_single_reason("ambiguous query")
print(result['probabilities'])
```

### Model Training Issues

```bash
# Check if healthcare data is available
ls -la data/reason_for_visit_data.xlsx

# Verify model training
python classifier/reason/train_reason.py

# Test inference after training
python classifier/reason/infer_reason.py
```

## Contributing

### Adding New Categories

1. Update `REASON_CATEGORIES` in `reason_classifier.py`, `infer_reason.py`, and `train_reason.py`
2. Update category mapping logic in `map_reason_to_category()`
3. Retrain the model with new categories
4. Update documentation and examples

### Improving Training Data

1. Add more real healthcare examples to the dataset
2. Improve keyword mapping for better categorization
3. Implement more sophisticated NLP techniques for category assignment

## License

This module is part of the health-query-classifier project and follows the same licensing terms.