from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from head import ClassifierHead
import os
import pandas as pd
from visualize import visualize_dataset
from sklearn.model_selection import train_test_split

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
        model_head = ClassifierHead(num_classes)
        model = SetFitModel(model_body, model_head)

    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Please ensure you have an internet connection and the transformers library installed.")
        raise RuntimeError("Failed to load the embedding model.")
    
    return model

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

    df.pop('Gender')
    df.pop('Age')
    df.pop('Duration')
    df.pop('Final Recommendation')

    df.rename(
        columns={
            'Symptoms': 'inquery',
            'Severity': 'severity'
        },
        inplace=True
    )
    return df

def main():
    df = get_dataset()
    df = preprocess(df)
    # visualize_dataset(df)
    labels = df['severity'].unique()

    X = df['inquery']
    y = df['severity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = get_model(len(labels))

    trainer = Trainer(
        model=model,
    )

    metrics = trainer.evaluate(test_dataset)
    print(metrics)

if __name__ == "__main__":
    main()
