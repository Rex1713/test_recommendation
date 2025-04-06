import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq

# --- Set up Llama 4 model from Groq ---
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
Settings.llm = Groq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))

# --- Set embedding model ---
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")

# --- Load the persisted index ---
persist_dir = "shl_index"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
index = load_index_from_storage(storage_context)

# --- Query the index ---
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("Which assessments support adaptive testing?")

# --- Print the response ---
print("\n🔎 Response:")
print(response)
