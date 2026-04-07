"""
src/ingestion/semantic_chunker.py

Envuelve el SemanticChunker de LangChain con los defaults del proyecto.
Usa un sentence-transformer multilingue optimizado para español.

El chunking semantico agrupa texto por significado y coherencia tematica,
en lugar de segmentar por tamano fijo de tokens.
"""
from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from src.embeddings import get_embeddings
from src.config import (
    CHUNK_BREAKPOINT_TYPE,
    CHUNK_BREAKPOINT_THRESHOLD,
)


def build_semantic_chunker() -> SemanticChunker:
    """
    Retorna una instancia configurada de SemanticChunker.

    breakpoint_threshold_type opciones:
        "percentile"         — divide cuando distancia > percentil N de todas las distancias
        "standard_deviation" — divide en media + N*std
        "interquartile"      — basado en IQR
    """
    chunker = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type=CHUNK_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=CHUNK_BREAKPOINT_THRESHOLD,
    )
    return chunker


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Divide una lista de Documents (nivel pagina) en chunks semanticos.
    Preserva y propaga la metadata del documento fuente en cada chunk.
    Agrega un campo 'chunk_index' a la metadata de cada chunk.
    """
    chunker = build_semantic_chunker()
    chunks: List[Document] = []
    for doc in documents:
        if not doc.page_content.strip():
            continue
        try:
            split_docs = chunker.create_documents(
                [doc.page_content],
                metadatas=[doc.metadata],
            )
            for idx, chunk in enumerate(split_docs):
                chunk.metadata["chunk_index"] = len(chunks) + idx
            chunks.extend(split_docs)
        except Exception as exc:
            # Algunas paginas pueden ser muy cortas para el chunker
            print(f"[WARN] Chunking fallido para {doc.metadata.get('source', '?')} "
                  f"p.{doc.metadata.get('page', '?')}: {exc}")
            # Fallback: usar el documento completo como un solo chunk
            doc.metadata["chunk_index"] = len(chunks)
            chunks.append(doc)
    print(f"[INFO] Generados {len(chunks)} chunks semanticos de {len(documents)} paginas.")
    return chunks
