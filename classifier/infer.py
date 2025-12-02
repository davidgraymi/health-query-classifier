from classifier.head import ClassifierHead
from classifier.utils import CATEGORIES, CHECKPOINT_PATH, DEVICE, get_models, CLASSIFIER_NAME, get_latest_checkpoint

import argparse
import pprint
import torch
from sentence_transformers import SentenceTransformer

def classifier_init(checkpoint_path: str | None = None, model_id: str | None = CLASSIFIER_NAME) -> (SentenceTransformer, ClassifierHead):
    if checkpoint_path:
        latest_checkpoint = get_latest_checkpoint(checkpoint_path)
        print(f"Loading checkpoint from {latest_checkpoint}")
        embedding_model, classifier = get_models(model_id=latest_checkpoint)
    else:
        embedding_model, classifier = get_models(model_id=model_id)

    return embedding_model, classifier

def predict_query(
    text: list[str],
    embedding_model: SentenceTransformer,
    classifier_head: ClassifierHead,
) -> dict:
    """
    Runs the full inference pipeline: Text -> Embedding -> Classification.
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
        predicted_indices = torch.argmax(probabilities, dim=1).unsqueeze(1)
        confidences = torch.gather(probabilities, dim=1, index=predicted_indices).squeeze().tolist()

        # Get the predicted label name
        predicted_labels = [CATEGORIES[i] for i in predicted_indices]

    return {
        'prediction': predicted_labels,
        'confidence': confidences,
        'probabilities': probabilities.cpu().squeeze().tolist()
    }

def test(local: bool = False):
    embedding_model, classifier = classifier_init(checkpoint_path=CHECKPOINT_PATH if local else None)

    queries = [
        "Hi! I'm having a really bad rash on my hands. I'm pretty sure it's my excema flairing up. Is there anythign stronger than aquaphor I can use on it?",
        "Hey is there any way I can get an appointment in the next month?",
        "Hey is there any way I can get an appointment in the next month with a doctor?",
        "I'm traveling to South America soon. Do I need to get any vaccines before I go?",
        "I have this rash that popped up today.",
        "How can I make this hosptial bill go away?",
        "I'm so confused do I have to cover the full cost of this operation?",
    ]

    pred = predict_query(
        text=queries,
        embedding_model=embedding_model,
        classifier_head=classifier,
    )

    pprint.pprint(pred, indent=4)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Inference on a classifier for triaging health queries"
    )
    ap.add_argument(
        "--local", action="store_true",
        help="Use local checkpoint"
    )
    args = ap.parse_args()

    test(local=args.local)
