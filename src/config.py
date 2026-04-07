"""
Central configuration — reads .env and exposes typed constants.
All path and credential references in the codebase must import from here.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- Project root ----------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# --- Document corpus -------------------------------------------------------
DOCS_DIR: Path = ROOT / "content" / "docs"

# --- Vector store ----------------------------------------------------------
CHROMA_PERSIST_DIR: Path = ROOT / ".chroma_db"
CHROMA_COLLECTION_NAME: str = "insolvencia_chunks"
EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_BREAKPOINT_TYPE: str = "percentile"   # for SemanticChunker
CHUNK_BREAKPOINT_THRESHOLD: float = 95.0

# --- Ontology / KG ---------------------------------------------------------
ONTOLOGY_PATH: Path = ROOT / "ontology" / "insolvencia.ttl"
GRAPHDB_ENDPOINT: str = os.getenv("GRAPHDB_ENDPOINT", "http://localhost:7200")
GRAPHDB_REPOSITORY: str = os.getenv("GRAPHDB_REPOSITORY", "insolvencia")
GRAPHDB_SPARQL_ENDPOINT: str = (
    f"{GRAPHDB_ENDPOINT}/repositories/{GRAPHDB_REPOSITORY}"
)
GRAPHDB_UPDATE_ENDPOINT: str = (
    f"{GRAPHDB_ENDPOINT}/repositories/{GRAPHDB_REPOSITORY}/statements"
)
ONTOLOGY_BASE_URI: str = "http://www.unal.edu.co/ontologies/insolvencia#"

# --- LLM -------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = 0.0

# --- LangSmith -------------------------------------------------------------
LANGCHAIN_TRACING_V2: str = "true"
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "insolvencia-kg-rag")

# --- Agent -----------------------------------------------------------------
MAX_RETRIES: int = 3
RETRIEVAL_K: int = 6       # number of chunks per retrieval call
MMR_LAMBDA: float = 0.5    # diversity vs. relevance balance
