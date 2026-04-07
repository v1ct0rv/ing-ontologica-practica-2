"""
src/tracing/langsmith_setup.py

Configura el tracing de LangSmith para todos los nodos de LangGraph.
Debe importarse y llamar configure_langsmith() ANTES de crear
cualquier objeto LangChain/LangGraph.

LangSmith registra automaticamente la ejecucion de cada nodo cuando
LANGCHAIN_TRACING_V2=true esta configurado. No se necesita
instrumentacion manual adicional.
"""
import os
from src.config import (
    LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT,
    LANGCHAIN_TRACING_V2,
)


def configure_langsmith() -> None:
    """
    Configura las variables de entorno requeridas por el SDK de LangSmith.
    Llamar una vez al inicio de la aplicacion (antes de construir chains).
    """
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    print(f"[LangSmith] Tracing habilitado. Proyecto: {LANGCHAIN_PROJECT}")


def get_run_url(run_id: str) -> str:
    """Retorna la URL del dashboard LangSmith para un run_id dado."""
    return f"https://smith.langchain.com/o/default/projects/{LANGCHAIN_PROJECT}/runs/{run_id}"


def is_tracing_enabled() -> bool:
    """Verifica si el tracing de LangSmith esta habilitado y configurado."""
    return (
        os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and bool(os.environ.get("LANGCHAIN_API_KEY", ""))
    )
