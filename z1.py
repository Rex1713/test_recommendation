import streamlit as st
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

def run_streamlit_app():
    st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
    st.title("🧠 SHL Assessment Recommender")

    # User input
    user_input = st.text_input("Enter a job description or a URL pointing to one:", "")

    if st.button("🔍 Find Relevant Assessments") and user_input:
        if user_input.startswith("http://") or user_input.startswith("https://"):
            query = extract_text_from_url(user_input)
            st.markdown("**🔍 Extracted job description from URL (preview):**")
            st.write(query[:500] + "..." if len(query) > 500 else query)
        else:
            query = user_input

        if not query:
            st.error("❌ No valid query found.")
            return

        # Load index
        csv_path = "rex.csv"
        persist_dir = "shl_index"

        if not os.path.exists(persist_dir):
            nodes = load_shl_data_with_metadata(csv_path)
            index = VectorStoreIndex(nodes)
            index.storage_context.persist(persist_dir=persist_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)

        # Run the query
        query_engine = index.as_query_engine(
            similarity_top_k=10,
            response_mode="compact_and_refine",
            response_synthesizer=CompactAndRefine()
        )
        response = query_engine.query(query)

        # Create table of results
        records = []
        for node in response.source_nodes:
            meta = node.node.metadata
            records.append({
                "Assessment Name": meta["assessment_name"],
                "Remote Support": meta["remote"],
                "Adaptive Support": meta["adaptive"],
                "Duration": "Untimed" if meta["duration_minutes"] == 9999 else f"{meta['duration_minutes']} mins",
                "Type": meta["type"],
                "URL": meta["url"]
            })

        if records:
            df = pd.DataFrame(records)
            
            # Keep Assessment Name and URL in separate columns
            df["Link"] = df["URL"].apply(lambda url: f"[Link]({url})")
            
            # Optionally drop the raw URL column if you only want the clickable link
            df.drop(columns=["URL"], inplace=True)

            st.markdown("### 📋 Top Recommended Assessments")
            st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        else:
            st.warning("No relevant assessments found.")


        # Show LLM output
        st.markdown("### 🧠 LLM-Synthesized Summary")
        st.markdown(response.response)


if __name__ == "__main__":
    run_streamlit_app()
