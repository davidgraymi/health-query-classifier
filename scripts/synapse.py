import os
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np

SYNAPSE_DATASET_URL = "https://figshare.com/ndownloader/files/57688621"
MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"
EMBEDDINGS_FILE = "symptom_embeddings.npy"

def get_model() -> SentenceTransformer:
    """
    Loads sentence-transformers/embeddinggemma-300m-medicalmodel with prompts
    """
    try:
        return SentenceTransformer(
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
            default_prompt_name=None,
        )
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print("Please ensure you have an internet connection and the transformers library installed.")
        raise RuntimeError("Failed to load the embedding model.")

def analyze_symptoms_with_gemma(df: pd.DataFrame, column_name='Symptoms', base_data_dir="data"):
    """
    Generates embeddings for symptoms using a fined tuned version of Google's Gemma model and visualizes them using T-SNE.
    """
    print("\n--- Semantic Clustering with sentence-transformers/embeddinggemma-300m-medical ---")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 1. Load Model and Tokenizer
    model = get_model()

    # 2. Sample Data
    symptoms_list = df[column_name].tolist()
    print(f"Analyzing a sample of {len(symptoms_list)} symptoms...")

    # 3. Generate Embeddings in batches to manage memory
    embeddings = model.encode(symptoms_list, prompt_name='clustering', show_progress_bar=True)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    # Save embeddings to file
    embeddings_path = os.path.join(base_data_dir, EMBEDDINGS_FILE)
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved to {embeddings_path}")

    # 4. Dimensionality Reduction (T-SNE)
    print("Applying T-SNE for 2D visualization (This may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    # Fit and transform the embeddings to 2D
    tsne_results = tsne.fit_transform(embeddings)

    # 5. Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=10, alpha=0.6)
    plt.title('T-SNE Visualization of Symptom Embeddings (Gemma 300M)')
    plt.xlabel('T-SNE Dimension 1')
    plt.ylabel('T-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.show()
    print("\nVisualization complete. Examine the plot for clusters.")

def visualize_data(df: pd.DataFrame):
    """
    Analyzes and visualizes the data for easy digestion.
    """

    # display basic info
    print("\nDataFrame Info:")
    df.info()

    # summary
    print("\nSummary Statistics for Numerical Columns:")
    print(df.describe().to_markdown(numalign="left", stralign="left"))

    print("\nValue Counts for Categorical Columns:")
    categorical_cols = df.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts().to_markdown())

    analyze_symptoms_with_gemma(df)

def download(base_data_dir="data") -> str:
    """
    Downloads the Synapse dataset and saves it locally.
    """
    os.makedirs(base_data_dir, exist_ok=True)
    dataset_path = os.path.join(base_data_dir, "synapse.csv")

    if not os.path.exists(dataset_path):
        print("Downloading dataset...")
        try:
            df = pd.read_csv(SYNAPSE_DATASET_URL)
            df.to_csv(dataset_path, index=False)
            print(f"Dataset downloaded and saved to {dataset_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
    else:
        print(f"Dataset already exists at {dataset_path}")

    return dataset_path

def main():
    """
    Main function to download, save, and visualize the dataset.
    """
    dataset_path = download()

    dtype_spec = {
        'Symptoms': 'object',
        'Gender': 'category',
        'Age': 'category',
        'Duration': 'category',
        'Severity': 'category',
        'Final Recommendation': 'category'
    }

    # load data for visualization
    try:
        df = pd.read_csv(dataset_path, dtype=dtype_spec)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return # Exit if load fails

    # visualize data
    visualize_data(df)

if __name__ == "__main__":
    main()