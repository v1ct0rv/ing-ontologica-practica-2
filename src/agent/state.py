"""
src/agent/state.py

Definicion TypedDict del estado del agente LangGraph.
Cada nodo lee y escribe a este estado compartido.
"""
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Estado compartido entre todos los nodos del grafo LangGraph."""

    # Hilo de mensajes (append-only via add_messages)
    messages: Annotated[List[BaseMessage], add_messages]

    # Pregunta original del usuario
    question: str

    # Sub-preguntas transformadas (HyDE o descompuestas)
    transformed_queries: List[str]

    # Ruta de recuperacion decidida por el router (DIRECT/HYDE/DECOMPOSE)
    route: str

    # Chunks de contexto recuperados del vector store
    context_chunks: List[str]

    # Resultados de consultas al Knowledge Graph
    kg_results: List[str]

    # Respuesta generada actual
    answer: str

    # Critica/reflexion del nodo critic
    critique: Optional[str]

    # Numero de iteraciones de reintento completadas
    retry_count: int

    # Si el nodo de reflexion aprobo la respuesta
    answer_approved: bool

    # Si se uso el fallback web
    used_web_fallback: bool

    # Flag para prevenir loops infinitos despues del web fallback
    web_fallback_used: bool

    # Fuentes citadas en la respuesta final
    sources: List[str]
