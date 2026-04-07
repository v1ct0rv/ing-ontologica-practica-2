"""
src/agent/nodes.py

Funciones de nodo para LangGraph implementando el patron ReAct + Reflecting.
Cada funcion tiene la firma: (state: AgentState) -> dict[str, Any]
El dict retornado se fusiona con el estado compartido.

Nodos:
  1. query_transform — Clasifica, descompone y/o aplica HyDE
  2. retrieve        — Busca en vector store y/o KG segun la ruta
  3. generate        — Sintetiza respuesta a partir del contexto
  4. reflect         — Evalua/critica la respuesta (aprueba o rechaza)
  5. web_fallback    — Busqueda web cuando se agotan reintentos
"""
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.config import LLM_MODEL, LLM_TEMPERATURE, MAX_RETRIES
from src.query.hyde import apply_hyde
from src.query.decomposer import decompose_query
from src.query.router import route_query


def _make_llm(temperature: float = LLM_TEMPERATURE) -> ChatOpenAI:
    """Crea una instancia del LLM con la configuracion del proyecto."""
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


# ---------------------------------------------------------------------------
# Nodo 1: query_transform
# Analiza la consulta y decide la estrategia de transformacion
# ---------------------------------------------------------------------------
def query_transform_node(state: AgentState) -> Dict[str, Any]:
    """
    Clasifica la pregunta (DIRECT/HYDE/DECOMPOSE), descompone si es necesario,
    y aplica HyDE para generar documentos hipoteticos que mejoren la busqueda.
    """
    llm = _make_llm()
    question = state["question"]

    # Clasificar la estrategia (barato, rapido)
    route = route_query(llm, question)

    # Descomponer en sub-preguntas si es DECOMPOSE
    if route == "DECOMPOSE":
        sub_questions = decompose_query(llm, question)
    else:
        sub_questions = [question]

    # Aplicar HyDE si la estrategia lo requiere
    if route == "HYDE":
        transformed = [apply_hyde(llm, q) for q in sub_questions]
    else:
        transformed = sub_questions

    return {
        "route": route,
        "transformed_queries": transformed,
        "messages": [AIMessage(content=(
            f"[QueryTransform] Estrategia: {route}. "
            f"Sub-preguntas: {len(sub_questions)}. "
            f"Queries transformadas: {len(transformed)}"
        ))],
    }


# ---------------------------------------------------------------------------
# Nodo 2: retrieve
# Busca en vector store y/o Knowledge Graph segun la ruta
# ---------------------------------------------------------------------------
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Ejecuta la recuperacion usando las queries transformadas.
    Combina resultados del vector store (busqueda semantica MMR)
    y del Knowledge Graph (consultas SPARQL).
    """
    from src.ingestion.vector_store import load_vector_store
    from src.kg.sparql_tools import query_kg_norms, query_kg_procedures, query_kg_entity

    queries = state["transformed_queries"]
    if not queries:
        queries = [state["question"]]

    chunks: List[str] = state.get("context_chunks", [])
    kg_results: List[str] = state.get("kg_results", [])
    sources: List[str] = state.get("sources", [])

    # Busqueda vectorial
    try:
        store = load_vector_store()
        for q in queries:
            docs = store.max_marginal_relevance_search(
                q, k=6, fetch_k=18, lambda_mult=0.5
            )
            for doc in docs:
                src = doc.metadata.get("source", "desconocido")
                pg = doc.metadata.get("page", "?")
                source_ref = f"{src}, p.{pg}"
                if source_ref not in sources:
                    sources.append(source_ref)
                chunks.append(
                    f"[Fuente: {source_ref}]\n{doc.page_content[:800]}"
                )
    except Exception as exc:
        chunks.append(f"[Error vector store: {exc}]")

    # Busqueda en Knowledge Graph
    for q in queries:
        q_lower = q.lower()
        try:
            if any(w in q_lower for w in ["ley", "decreto", "norma", "artículo", "articulo"]):
                result = query_kg_norms.invoke(q)
                if result and "No se encontraron" not in result:
                    kg_results.append(result)

            if any(w in q_lower for w in ["empresa", "deudor", "reorganiz", "liquidac", "procedimiento"]):
                result = query_kg_procedures.invoke(q)
                if result and "No se encontraron" not in result:
                    kg_results.append(result)
        except Exception:
            pass  # KG queries son complementarias, no criticas

    return {
        "context_chunks": chunks,
        "kg_results": kg_results,
        "sources": sources,
        "messages": [AIMessage(content=(
            f"[Retrieve] {len(chunks)} chunks vectoriales, "
            f"{len(kg_results)} resultados KG"
        ))],
    }


# ---------------------------------------------------------------------------
# Nodo 3: generate
# Sintetiza respuesta a partir del contexto recuperado
# ---------------------------------------------------------------------------
_GENERATE_SYSTEM = """Eres un experto en derecho de insolvencia empresarial colombiana.
Responde la pregunta del usuario usando ÚNICAMENTE la información proporcionada
en el contexto. Si el contexto no es suficiente, indícalo explícitamente.

Reglas:
- Cita siempre las fuentes (nombre del documento y página) entre corchetes
- Usa terminología legal precisa
- Estructura la respuesta de forma clara y organizada
- Si hay información del Grafo de Conocimiento, úsala para enriquecer la respuesta"""


def generate_node(state: AgentState) -> Dict[str, Any]:
    """Genera una respuesta sintetizando el contexto vectorial y del KG."""
    llm = _make_llm()

    # Limitar contexto para no exceder ventana del LLM
    context_text = "\n\n---\n\n".join(state.get("context_chunks", []))[:8000]
    kg_text = "\n\n".join(state.get("kg_results", []))[:2000]

    full_context = f"DOCUMENTOS RECUPERADOS:\n{context_text}"
    if kg_text:
        full_context += f"\n\nINFORMACION DEL GRAFO DE CONOCIMIENTO:\n{kg_text}"

    # Si hay critica previa, incluirla como guia
    critique = state.get("critique", "")
    critique_hint = ""
    if critique and not state.get("answer_approved", False):
        critique_hint = (
            f"\n\nNOTA: Una evaluación previa indicó: {critique}\n"
            "Por favor, aborda estos puntos en tu respuesta."
        )

    messages = [
        SystemMessage(content=_GENERATE_SYSTEM),
        HumanMessage(content=(
            f"CONTEXTO:\n{full_context}\n\n"
            f"PREGUNTA: {state['question']}"
            f"{critique_hint}"
        )),
    ]
    response = llm.invoke(messages)
    return {
        "answer": response.content,
        "messages": [AIMessage(content=f"[Generate] Respuesta generada ({len(response.content)} chars)")],
    }


# ---------------------------------------------------------------------------
# Nodo 4: reflect (critic)
# Evalua la respuesta y decide si es aceptable
# ---------------------------------------------------------------------------
_REFLECT_SYSTEM = """Eres un crítico experto en derecho de insolvencia empresarial colombiano.
Evalúa la respuesta proporcionada en términos de:

1. **Fidelidad**: ¿La respuesta se basa en el contexto proporcionado sin inventar hechos?
2. **Completitud**: ¿Responde la pregunta del usuario de forma razonable?
3. **Precisión legal**: ¿Menciona normas o conceptos jurídicos relevantes?
4. **Fuentes**: ¿Indica al menos una fuente de donde proviene la información?

IMPORTANTE: Sé pragmático. Una respuesta que aborde la pregunta con información del contexto
y cite al menos una fuente es APROBADA. Solo rechaza si la respuesta es claramente incorrecta,
no responde la pregunta, o no tiene ninguna fuente.

Si la respuesta ES satisfactoria, responde EXACTAMENTE: APROBADO
Si NO es satisfactoria, responde EXACTAMENTE: RECHAZADO: <explicación breve de qué falta>"""


def reflect_node(state: AgentState) -> Dict[str, Any]:
    """
    Evalua la respuesta generada. Si es aprobada, el flujo termina.
    Si es rechazada, incrementa retry_count y el flujo vuelve a retrieve.
    """
    llm = _make_llm(temperature=0.0)

    context_preview = " ".join(state.get("context_chunks", []))[:2000]

    messages = [
        SystemMessage(content=_REFLECT_SYSTEM),
        HumanMessage(content=(
            f"PREGUNTA ORIGINAL: {state['question']}\n\n"
            f"RESPUESTA GENERADA:\n{state['answer']}\n\n"
            f"CONTEXTO DISPONIBLE (resumen):\n{context_preview}"
        )),
    ]
    critique = llm.invoke(messages).content.strip()
    approved = critique.upper().startswith("APROBADO")

    new_retry = state.get("retry_count", 0)
    if not approved:
        new_retry += 1

    return {
        "critique": critique,
        "answer_approved": approved,
        "retry_count": new_retry,
        "messages": [AIMessage(content=(
            f"[Reflect] {'APROBADO' if approved else f'RECHAZADO (intento {new_retry}/{MAX_RETRIES})'}: "
            f"{critique[:200]}"
        ))],
    }


# ---------------------------------------------------------------------------
# Nodo 5: web_fallback
# Busqueda web cuando se agotan los reintentos
# ---------------------------------------------------------------------------
def web_fallback_node(state: AgentState) -> Dict[str, Any]:
    """
    Ejecuta busqueda web via DuckDuckGo y agrega resultados al contexto.
    Establece web_fallback_used=True para prevenir que la reflexion
    dispare otro ciclo de reintento despues del fallback.
    """
    from src.agent.web_fallback import search_web

    results = search_web(state["question"])
    existing_chunks = state.get("context_chunks", [])

    return {
        "context_chunks": existing_chunks + [f"[FUENTE WEB]\n{results}"],
        "used_web_fallback": True,
        "web_fallback_used": True,
        "messages": [AIMessage(content="[WebFallback] Contexto web complementario recuperado.")],
    }
