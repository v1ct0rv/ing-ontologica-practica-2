"""
scripts/generate_traces.py

Genera 3 trazas en LangSmith para el informe técnico:
  1. Caso exitoso (aprobado en primer intento)
  2. Caso de reintento (critic rechaza, vuelve a retrieve)
  3. Caso de web fallback (3 rechazos → búsqueda web)

Uso: python scripts/generate_traces.py
Requiere: OPENAI_API_KEY y LANGCHAIN_API_KEY en .env
"""
import sys
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracing.langsmith_setup import configure_langsmith
configure_langsmith()

from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.agent import nodes as nodes_module


# ============================================================================
# Caso 1: Exitoso (critic pragmático normal)
# ============================================================================
def run_case_1():
    print("=" * 60)
    print("CASO 1: Aprobado en primer intento")
    print("=" * 60)

    graph = build_graph()
    state = _make_initial_state("¿Qué es la cesación de pagos según la Ley 1116 de 2006?")
    result = graph.invoke(state)
    _print_result(result)


# ============================================================================
# Caso 2: Reintento (critic estricto temporal)
# ============================================================================
def run_case_2():
    print("\n" + "=" * 60)
    print("CASO 2: Reintento (critic rechaza 1 vez, luego aprueba)")
    print("=" * 60)

    from langchain_core.messages import AIMessage
    from langgraph.graph import StateGraph, START, END
    from src.agent.nodes import query_transform_node, retrieve_node, generate_node, web_fallback_node
    from src.agent.state import AgentState
    from src.config import MAX_RETRIES

    # Reflect custom: rechaza 1 vez, aprueba la segunda
    def reflect_once_then_approve(state):
        retry = state.get("retry_count", 0)
        if retry == 0:
            return {
                "critique": "RECHAZADO: La respuesta no cita artículos específicos con su texto literal. Falta mencionar el artículo 9 sobre cesación de pagos.",
                "answer_approved": False,
                "retry_count": 1,
                "messages": [AIMessage(content="[Reflect] RECHAZADO (intento 1/3): Falta citar artículos específicos.")],
            }
        else:
            return {
                "critique": "APROBADO: La respuesta mejoró y cita fuentes correctamente.",
                "answer_approved": True,
                "retry_count": retry,
                "messages": [AIMessage(content="[Reflect] APROBADO en segundo intento.")],
            }

    def should_retry(state):
        if state.get("web_fallback_used", False):
            return "end"
        if state.get("answer_approved", False):
            return "end"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "web_fallback"
        return "retrieve"

    # Construir grafo custom con el reflect parcheado
    builder = StateGraph(AgentState)
    builder.add_node("query_transform", query_transform_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("reflect", reflect_once_then_approve)
    builder.add_node("web_fallback", web_fallback_node)
    builder.add_edge(START, "query_transform")
    builder.add_edge("query_transform", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "reflect")
    builder.add_conditional_edges("reflect", should_retry, {
        "end": END, "web_fallback": "web_fallback", "retrieve": "retrieve",
    })
    builder.add_edge("web_fallback", "generate")
    graph = builder.compile()

    state = _make_initial_state(
        "¿Cuáles son los requisitos para iniciar un proceso de reorganización empresarial?"
    )
    result = graph.invoke(state)
    _print_result(result)


# ============================================================================
# Caso 3: Web fallback (critic siempre rechaza → 3 retries → web)
# ============================================================================
def run_case_3():
    print("\n" + "=" * 60)
    print("CASO 3: Web Fallback (3 rechazos → búsqueda web)")
    print("=" * 60)

    from langchain_core.messages import AIMessage
    from langgraph.graph import StateGraph, START, END
    from src.agent.nodes import query_transform_node, retrieve_node, generate_node, web_fallback_node
    from src.agent.state import AgentState
    from src.config import MAX_RETRIES

    # Reflect que SIEMPRE rechaza
    def reflect_always_reject(state):
        retry = state.get("retry_count", 0) + 1
        return {
            "critique": f"RECHAZADO: La respuesta requiere información más actualizada y detallada (intento {retry}/{MAX_RETRIES}).",
            "answer_approved": False,
            "retry_count": retry,
            "messages": [AIMessage(content=f"[Reflect] RECHAZADO (intento {retry}/{MAX_RETRIES})")],
        }

    def should_retry(state):
        if state.get("web_fallback_used", False):
            return "end"
        if state.get("answer_approved", False):
            return "end"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "web_fallback"
        return "retrieve"

    builder = StateGraph(AgentState)
    builder.add_node("query_transform", query_transform_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("reflect", reflect_always_reject)
    builder.add_node("web_fallback", web_fallback_node)
    builder.add_edge(START, "query_transform")
    builder.add_edge("query_transform", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "reflect")
    builder.add_conditional_edges("reflect", should_retry, {
        "end": END, "web_fallback": "web_fallback", "retrieve": "retrieve",
    })
    builder.add_edge("web_fallback", "generate")
    graph = builder.compile()

    state = _make_initial_state(
        "¿Qué establece la Ley 2445 de 2025 sobre insolvencia de persona natural?"
    )
    result = graph.invoke(state)
    _print_result(result)


# ============================================================================
# Helpers
# ============================================================================
def _make_initial_state(question: str) -> AgentState:
    return {
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


def _print_result(result):
    print(f"\nEstrategia: {result.get('route', 'N/A')}")
    print(f"Reintentos: {result.get('retry_count', 0)}")
    print(f"Web Fallback: {'Sí' if result.get('used_web_fallback') else 'No'}")
    print(f"Crítica: {str(result.get('critique', ''))[:150]}")
    print(f"\nRespuesta ({len(result.get('answer', ''))} chars):")
    print(result.get("answer", "")[:500])
    if result.get("sources"):
        print(f"\nFuentes: {', '.join(result['sources'][:5])}")
    print()


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        caso = sys.argv[1]
        if caso == "1":
            run_case_1()
        elif caso == "2":
            run_case_2()
        elif caso == "3":
            run_case_3()
        else:
            print(f"Caso no válido: {caso}. Use 1, 2 o 3.")
    else:
        print("Generando 3 trazas en LangSmith...")
        print("Revisa https://smith.langchain.com después de ejecutar.\n")
        run_case_1()
        run_case_2()
        run_case_3()
        print("=" * 60)
        print("¡Listo! Revisa las 3 trazas en LangSmith.")
        print("=" * 60)
