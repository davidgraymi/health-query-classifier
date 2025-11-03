from sentence_transformers import SentenceTransformer
from setfit import SetFitModel
from head import ClassifierHead
import os
import pandas as pd
from visualize import visualize_dataset

SYNAPSE_DATASET_URL = "https://figshare.com/ndownloader/files/57688621"
MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"

def get_model():
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
        model_head = ClassifierHead(5)
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

def transform_to_nlp_classification(row):
    row['Symptoms'] = f"{row['Symptoms']} {row['Gender']} {row['Age']} {row['Duration']}"
    return row

def main():
    df = get_dataset()
    model = get_model()
    visualize_dataset(df)

if __name__ == "__main__":
    main()
