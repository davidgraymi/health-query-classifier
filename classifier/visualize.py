import pandas as pd

def visualize_dataset(df: pd.DataFrame):
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

# def analyze_symptoms_with_gemma(df: pd.DataFrame, column_name='Symptoms', base_data_dir="data"):
#     """
#     Generates embeddings for symptoms using a fined tuned version of Google's Gemma model and visualizes them using T-SNE.
#     """
#     print("\n--- Semantic Clustering with sentence-transformers/embeddinggemma-300m-medical ---")

#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     embeddings_path = os.path.join(base_data_dir, EMBEDDINGS_FILE)
    
#     model = get_model()
#     embeddings = get_embeddings(model, embeddings_path, df[column_name].tolist())

#     print("Applying T-SNE for 2D visualization (This may take a minute)...")
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#     # Fit and transform the embeddings to 2D
#     tsne_results = tsne.fit_transform(embeddings)

#     # 5. Visualization
#     plt.figure(figsize=(10, 8))
#     plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=10, alpha=0.6)
#     plt.title('T-SNE Visualization of Symptom Embeddings (Gemma 300M)')
#     plt.xlabel('T-SNE Dimension 1')
#     plt.ylabel('T-SNE Dimension 2')
#     plt.grid(True, alpha=0.3)
#     plt.show()
#     print("\nVisualization complete. Examine the plot for clusters.")