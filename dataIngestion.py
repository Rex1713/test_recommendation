import os
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from llama_index.core.settings import Settings

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")


def load_shl_data_with_metadata(csv_path: str):
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        # Extract values
        assessment_name = str(row['Assessment Name'])
        description = str(row['Description'])
        assessment_type = str(row['Types']).strip()
        raw_duration = str(row['Assessment Length (minutes)']).strip()
        remote = str(row['Remote Support']).strip()
        adaptive = str(row['Adaptive Support']).strip()
        job_levels = str(row['Job Levels']).strip()
        url = str(row['URL']).strip()

        # Normalize duration: convert to readable format
        try:
            minutes = int(float(raw_duration))
            if minutes == 9999:
                duration_clean = "Untimed"
            elif minutes == -1:
                duration_clean = "Variable"
            else:
                duration_clean = f"{minutes} minutes"
        except ValueError:
            duration_clean = "Variable"
            minutes = -1

        # Full node text
        text = f"""
        Assessment: {assessment_name}
        Description: {description}
        Type: {assessment_type}
        Duration: {duration_clean}
        Remote: {remote}
        Adaptive: {adaptive}
        Job Levels: {job_levels}
        URL: {url}
        """

        # Metadata for filtering
        metadata = {
            "assessment_name": assessment_name,
            "type": assessment_type,
            "duration_minutes": minutes,
            "remote": remote,
            "adaptive": adaptive,
            "job_levels": job_levels,
            "url": url
        }

        node = TextNode(text=text.strip(), metadata=metadata)
        documents.append(node)

    return documents

def main():
    # Load CSV
    csv_path = "rex.csv"  # Change this path if needed
    print("📄 Loading data from:", csv_path)

    nodes = load_shl_data_with_metadata(csv_path)
    print(f"✅ Loaded {len(nodes)} assessments.")

    # Build the vector index
    index = VectorStoreIndex(nodes)
    print("📦 Index built.")

    # Save the index to disk
    index.storage_context.persist(persist_dir="shl_index")
    print("💾 Index saved to 'shl_index' folder.")

    # Print one sample for validation
    print("\n🔍 Sample node:")
    print(nodes[0].text)
    print("📎 Metadata:", nodes[0].metadata)

if __name__ == "__main__":
    main()
