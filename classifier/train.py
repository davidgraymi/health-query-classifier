from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from head import ClassifierHead
import os
import pandas as pd
from visualize import visualize_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

SYNAPSE_DATASET_URL = "https://figshare.com/ndownloader/files/57688621"
MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"

def get_model(num_classes: int):
    try:
        model_body = SentenceTransformer(
            MODEL_NAME,
            prompts={
                'retrieval (query)': 'task: search result | query: ',
                'retrieval (document)': 'title: {title | "none"} | text: ',
                'qa': 'task: question answering | query: ',
                'fact verification': 'task: fact checking | query: ',
                'classification': 'task: classification | query: ',
                'clustering': 'task: clustering | query: ',
                'semantic similarity': 'task: sentence similarity | query: ',
                'code retrieval': 'task: code retrieval | query: '
            },
            default_prompt_name='classification',
        )
        # Freeze weights of embedding model
        model_head = ClassifierHead(num_classes)
        model = SetFitModel(model_body, model_head)
        model.freeze("body")

    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Please ensure you have an internet connection and the transformers library installed.")
        raise RuntimeError("Failed to load the embedding model.")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print("MPS device not found. Using CPU.")
        device = torch.device("cpu")
    
    return model.to(device)

def get_dataset(base_data_dir="data") -> pd.DataFrame:
    """
    Downloads the Synapse dataset, saves it locally, returns a pandas dataframe.
    """
    os.makedirs(base_data_dir, exist_ok=True)
    dataset_path = os.path.join(base_data_dir, "synapse.csv")
    dtype_spec = {
        'Symptoms': 'object',
        'Gender': 'category',
        'Age': 'category',
        'Duration': 'category',
        'Severity': 'category',
        'Final Recommendation': 'category'
    }

    try:
        if not os.path.exists(dataset_path):
            print("Downloading dataset...")
            df = pd.read_csv(SYNAPSE_DATASET_URL, dtype=dtype_spec)
            df.to_csv(dataset_path, index=False)
            print(f"Dataset downloaded and saved to {dataset_path}")
        else:
            print("Found dataset locally...")
            df = pd.read_csv(dataset_path, dtype=dtype_spec)

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise Exception

    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['Symptoms'] = (
        df['Symptoms'].astype(str) + ' ' + 
        df['Gender'].astype(str) + ' ' + 
        df['Age'].astype(str) + ' ' + 
        df['Duration'].astype(str)
    )

    df['Severity'] = df['Severity'].cat.codes

    df.pop('Gender')
    df.pop('Age')
    df.pop('Duration')
    df.pop('Final Recommendation')

    df.rename(
        columns={
            'Symptoms': 'text',
            'Severity': 'label'
        },
        inplace=True
    )
    return df

def main():
    df = get_dataset()
    df = preprocess(df)
    labels = df['label'].unique()
    model = get_model(len(labels))

    # Preform 1000 sample evaluation
    train, test = train_test_split(
        df.sample(5000), test_size=0.3, random_state=42
    )

    train = Dataset.from_pandas(train)
    test = Dataset.from_pandas(test)

    args = TrainingArguments(
        output_dir="classifier/checkpoints",
        # Explicitly set body_epochs to 0 to skip the contrastive fine-tuning stage
        # This prevents the trainer from attempting to train the frozen 'body'.
        # Set the classification head epochs higher, as this is the only training that will run
        num_epochs=(0, 16),
        eval_strategy='epoch',
        eval_steps=500,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train,
        eval_dataset=test,
        metric='accuracy',
        column_mapping={"text": "text", "label": "label"},
        args=args,
    )

    trainer.train()

    metrics = trainer.evaluate(test)
    print(f"After training: {metrics}")

if __name__ == "__main__":
    main()
