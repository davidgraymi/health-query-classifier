from head import ClassifierHead
from utils import CATEGORIES, CHECKPOINT_PATH, DATETIME_FORMAT, DEVICE, get_models

from datetime import datetime
import os
import pprint
import torch
from sentence_transformers import SentenceTransformer

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


def test():
    latest = None
    path = ""
    for d in os.listdir(CHECKPOINT_PATH):
        t = datetime.strptime(d, DATETIME_FORMAT)
        if latest is None or t > latest:
            latest = t
            path = f"{CHECKPOINT_PATH}/{d}/final.pth"
            print("Loading checkpoint from ", path)

    state_dict = torch.load(path, weights_only=True)

    embedding_model, classifier = get_models()
    classifier.load_state_dict(state_dict)

    queries = [
        "Hi! I'm having a really bad rash on my hands. I'm pretty sure it's my excema flairing up. Is there anythign stronger than aquaphor I can use on it?",
        "Hey is there any way I can get an appointment in the next month?",
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
    test()