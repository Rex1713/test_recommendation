import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load Data
df = pd.read_csv("final.csv")

# Combine necessary fields for embedding
columns_to_combine = [
    "Assessment Name", "URL", "Remote Support", "Adaptive Support",
    "Types", "Description", "Job Levels", "Languages"
]
df["combined_text"] = df[columns_to_combine].apply(lambda row: " | ".join(row.values.astype(str)), axis=1)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, "shl_index.faiss")
df[["Assessment Name", "URL", "Remote Support", "Adaptive Support", "Types", "Description", "Job Levels", "Languages"]].to_pickle("shl_metadata.pkl")

print("Index and metadata saved successfully.")

# Sample search function
def search_shl(query, top_k=5):
    query_embedding = model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, top_k)
    results = df.iloc[I[0]].copy()
    return results[["Assessment Name", "URL", "Remote Support", "Adaptive Support", "Types", "Description", "Job Levels", "Languages"]]

# Example usage
if __name__ == "__main__":
    query = "sales manager for B2B software"
    top_results = search_shl(query, top_k=5)
    print("\nTop Recommendations:")
    print(top_results.to_markdown(index=False))
