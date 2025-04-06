import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
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

# --- Helper: Extract job description text from URL ---
def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
        return text.strip()
    except Exception as e:
        return f"Error extracting content from URL: {str(e)}"

# --- Load SHL assessments into nodes with metadata ---
def load_shl_data_with_metadata(csv_path: str):
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        assessment_name = str(row['Assessment Name'])
        description = str(row['Description'])
        assessment_type = str(row['Types']).strip()
        raw_duration = str(row['Assessment Length (minutes)']).strip()
        remote = str(row['Remote Support']).strip()
        adaptive = str(row['Adaptive Support']).strip()
        job_levels = str(row['Job Levels']).strip()
        url = str(row['URL']).strip()

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

# --- Main application ---
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

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

    # --- Accept user input (either URL or text) ---
    user_input = input("📝 Enter your query or job description URL: ").strip()

    if user_input.startswith("http://") or user_input.startswith("https://"):
        query = extract_text_from_url(user_input)
        print("\n🔍 Extracted job description from URL (preview):")
        print(query[:500] + "..." if len(query) > 500 else query)
    else:
        query = user_input

    if not query:
        print("❌ No valid query found.")
        return

    # --- Run the query ---
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        response_mode="compact_and_refine",
        response_synthesizer=CompactAndRefine()
    )
    response = query_engine.query(query)

    # --- Display top 10 recommendations in tabular form ---
    print("\n🔍 Top 5 Retrieved Nodes:")
    for i, node in enumerate(response.source_nodes):
        print(f"\nResult #{i+1}")
        print(node.node.text)
        # print("📎 Metadata:", node.node.metadata)


    print("\n🔎 Query Response:")
    print(response)


if __name__ == "__main__":
    main()
