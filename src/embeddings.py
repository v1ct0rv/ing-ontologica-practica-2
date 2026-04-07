"""
src/embeddings.py

Singleton para el modelo de embeddings. Se carga una sola vez
y silencia los mensajes de LOAD REPORT de safetensors (que se
imprimen desde código nativo C/Rust y no se capturan con
sys.stdout ni warnings).
"""
import os
import sys

from src.config import EMBEDDING_MODEL

_embeddings = None


def get_embeddings():
    """Retorna la instancia singleton del modelo de embeddings."""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        # Redirigir file descriptors nativos (captura output de C/Rust)
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        finally:
            # Restaurar file descriptors
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            os.close(devnull)
    return _embeddings
