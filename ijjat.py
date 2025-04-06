import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq
from llama_index.core.response_synthesizers import CompactAndRefine

load_dotenv()

# --- Set up Llama 4 model from Groq ---
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
Settings.llm = Groq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))

# --- Set embedding model ---
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")

def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        page_text = "\n".join([p.get_text() for p in paragraphs if p.get_text(strip=True)])
        return page_text[:2000]  # limit characters for safety
    except Exception as e:
        print(f"❌ Failed to fetch or parse URL: {e}")
        return ""

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
            "duration": duration_clean,
            "remote": remote,
            "adaptive": adaptive,
            "job_levels": job_levels,
            "url": url
        }

        node = TextNode(text=text.strip(), metadata=metadata)
        documents.append(node)

    return documents

def display_results_table(nodes):
    rows = []
    for node in nodes:
        meta = node.node.metadata
        name = meta.get("assessment_name", "")
        url = meta.get("url", "")
        name_link = f"[{name}]({url})" if url else name

        rows.append({
            "Assessment Name": name_link,
            "Type": meta.get("type", ""),
            "Duration": meta.get("duration", "Variable"),
            "Remote Support": meta.get("remote", ""),
            "Adaptive Support": meta.get("adaptive", ""),
            "Job Levels": meta.get("job_levels", "")
        })

    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', None)
    print("\n📊 Recommended Assessments:\n")
    print(df.to_markdown(index=False))

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

    # --- Get input ---
    input_query = input("\n🔍 Enter a job description (or URL):\n").strip()

    if input_query.startswith("http://") or input_query.startswith("https://"):
        input_query = extract_text_from_url(input_query)

    if not input_query:
        print("❗ No input provided. Exiting.")
        return

    # --- Query the index ---
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        response_mode="compact_and_refine",
        response_synthesizer=CompactAndRefine()
    )
    response = query_engine.query(input_query)

    # --- Display results ---
    display_results_table(response.source_nodes)

    # --- LLM Final Response ---
    print("\n🧠 LLM Final Response:\n")
    print(response.response)

if __name__ == "__main__":
    main()
