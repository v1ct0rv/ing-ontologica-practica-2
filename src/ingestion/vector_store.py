"""
src/ingestion/vector_store.py

Construye o carga un vector store ChromaDB a partir de chunks semanticos.
Expone un retriever con busqueda MMR (Maximal Marginal Relevance).
"""
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    RETRIEVAL_K,
    MMR_LAMBDA,
)
from src.embeddings import get_embeddings


def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Indexa chunks en ChromaDB y persiste a disco.
    Seguro llamar multiples veces — usa la coleccion existente si ya existe.
    """
    embeddings = get_embeddings()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )
    count = store._collection.count()
    print(f"[INFO] Vector store construido con {count} documentos.")
    return store


def load_vector_store() -> Chroma:
    """Carga una coleccion ChromaDB ya construida desde disco."""
    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            "Vector store no encontrado. Ejecute build_vector_store() primero."
        )
    embeddings = get_embeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


def get_mmr_retriever(
    store: Optional[Chroma] = None,
    k: int = RETRIEVAL_K,
    lambda_mult: float = MMR_LAMBDA,
) -> VectorStoreRetriever:
    """
    Retorna un retriever usando busqueda MMR (Maximal Marginal Relevance).
    Busca 2*k candidatos y selecciona k resultados diversos.

    Parametros:
        store: instancia de Chroma (si None, carga desde disco)
        k: numero de documentos a retornar
        lambda_mult: balance entre relevancia (1.0) y diversidad (0.0)
    """
    if store is None:
        store = load_vector_store()
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 3,
            "lambda_mult": lambda_mult,
        },
    )


def run_ingestion_pipeline() -> Chroma:
    """Pipeline completo: cargar PDFs -> chunking semantico -> indexar. Se ejecuta una vez."""
    from src.ingestion.pdf_loader import load_corpus
    from src.ingestion.semantic_chunker import chunk_documents

    docs = load_corpus()
    chunks = chunk_documents(docs)
    return build_vector_store(chunks)
