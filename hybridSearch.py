import os
import pandas as pd
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq
from llama_index.core.response_synthesizers import CompactAndRefine
from dotenv import load_dotenv
load_dotenv()


# --- Set up Llama 4 model from Groq ---
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
Settings.llm = Groq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))

# --- Set embedding model ---
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

        # Normalize duration
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
    csv_path = "rex.csv"
    persist_dir = "shl_index"

    if not os.path.exists(persist_dir):
        print("📄 Creating new index from:", csv_path)
        nodes = load_shl_data_with_metadata(csv_path)
        print(f"✅ Loaded {len(nodes)} assessments.")

        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=persist_dir)
        print("💾 Index saved to 'shl_index' folder.")
    else:
        print("📦 Loading existing index from disk...")

    # --- Load persisted index ---
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

    # --- Hybrid Search: Filter by Metadata First ---
    # --- Hybrid Search: Filter by Metadata First ---
    print("🧠 Performing Hybrid Search (metadata + vector)...")

    # Load all nodes
    all_nodes = list(index.docstore.docs.values())

    # Skill keywords
    required_skills = ["python", "sql", "javascript"]

    filtered_nodes = []
    for node in all_nodes:
        metadata = node.metadata
        text = node.text.lower()

        duration_ok = metadata.get("duration_minutes", 9999) <= 60
        job_level_ok = "mid" in str(metadata.get("job_levels", "")).lower()
        skills_ok = all(skill in text for skill in required_skills)

        if duration_ok and job_level_ok and skills_ok:
            filtered_nodes.append(node)

    print(f"✅ Filtered down to {len(filtered_nodes)} relevant assessments.")


    # --- Vector search only within filtered nodes ---
    hybrid_index = VectorStoreIndex(filtered_nodes)

    query_engine = hybrid_index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact_and_refine",
        response_synthesizer=CompactAndRefine()
    )

    response = query_engine.query(
        "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. "
        "Need an assessment package that can test all skills with max duration of 60 minutes"
    )

    # --- Output sample + result ---
    print("\n🔍 Top 5 Retrieved Nodes:")
    for i, node in enumerate(response.source_nodes):
        print(f"\nResult #{i+1}")
        print(node.node.text)

    print("\n🔎 Query Response:")
    print(response)


if __name__ == "__main__":
    main()
