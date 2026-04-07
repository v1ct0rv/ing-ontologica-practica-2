"""
src/agent/graph.py

Ensambla el StateGraph de LangGraph implementando el flujo completo
ReAct + Reflecting con web fallback.

Flujo:
  START
    -> query_transform   (clasifica, descompone, aplica HyDE)
    -> retrieve           (vector store MMR + Knowledge Graph SPARQL)
    -> generate           (sintetiza respuesta con LLM)
    -> reflect            (evalua/critica la respuesta)
    -> [aprobada?]
        SI  -> END
        NO  -> [retry_count >= MAX_RETRIES?]
            SI  -> web_fallback -> generate -> END
            NO  -> retrieve (loop de reintento)
"""
from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes import (
    query_transform_node,
    retrieve_node,
    generate_node,
    reflect_node,
    web_fallback_node,
)
from src.config import MAX_RETRIES


def should_retry(state: AgentState) -> str:
    """
    Edge condicional despues de reflect:
      - "end"          : respuesta aprobada O ya se uso web fallback
      - "web_fallback" : agotados los reintentos, buscar en web
      - "retrieve"     : reintentar con nuevo ciclo de recuperacion
    """
    # Si ya se uso web fallback, terminar (previene loop infinito)
    if state.get("web_fallback_used", False):
        return "end"

    # Si la respuesta fue aprobada, terminar
    if state.get("answer_approved", False):
        return "end"

    # Si se agotaron los reintentos, ir a web fallback
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "web_fallback"

    # Sino, reintentar recuperacion
    return "retrieve"


def build_graph() -> StateGraph:
    """
    Construye y compila el grafo del agente.
    Retorna el grafo compilado listo para invocar.
    """
    builder = StateGraph(AgentState)

    # Agregar nodos
    builder.add_node("query_transform", query_transform_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("web_fallback", web_fallback_node)

    # Edges del flujo principal
    builder.add_edge(START, "query_transform")
    builder.add_edge("query_transform", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "reflect")

    # Edge condicional despues de reflexion
    builder.add_conditional_edges(
        "reflect",
        should_retry,
        {
            "end": END,
            "web_fallback": "web_fallback",
            "retrieve": "retrieve",
        },
    )

    # Despues de web fallback, generar respuesta final y terminar
    builder.add_edge("web_fallback", "generate")

    graph = builder.compile()
    return graph


def run_query(question: str) -> dict:
    """
    Punto de entrada principal: invoca el grafo del agente con una pregunta.

    Retorna dict con:
      - answer: str (respuesta final)
      - sources: List[str] (fuentes citadas)
      - used_web_fallback: bool
      - retry_count: int
      - critique: str (ultima critica)
    """
    graph = build_graph()

    initial_state: AgentState = {
        "messages": [],
        "question": question,
        "transformed_queries": [],
        "route": "",
        "context_chunks": [],
        "kg_results": [],
        "answer": "",
        "critique": None,
        "retry_count": 0,
        "answer_approved": False,
        "used_web_fallback": False,
        "web_fallback_used": False,
        "sources": [],
    }

    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "used_web_fallback": final_state.get("used_web_fallback", False),
        "retry_count": final_state.get("retry_count", 0),
        "critique": final_state.get("critique", ""),
        "route": final_state.get("route", ""),
    }
