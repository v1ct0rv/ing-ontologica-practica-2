"""
src/agent/tools.py

Ensambla la lista completa de herramientas disponibles para el agente ReAct:
  1. vector_search   — Busqueda MMR en ChromaDB
  2. kg_procedures   — Consulta KG para procedimientos de insolvencia
  3. kg_norms        — Consulta KG para normas legales
  4. kg_creditors    — Consulta KG para tipos de acreedor
  5. kg_entity       — Consulta KG para propiedades de una entidad
  6. sparql_query    — SPARQL SELECT arbitrario sobre ontologia local
  7. web_search      — DuckDuckGo fallback (activado despues de 3 reintentos)
"""
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore

from src.kg.sparql_tools import KG_TOOLS


def build_vector_search_tool(vector_store: VectorStore):
    """Crea una herramienta de busqueda vectorial con MMR sobre el corpus."""

    @tool
    def vector_search(query: str) -> str:
        """
        Busca en el corpus de documentos de insolvencia empresarial usando
        similitud semantica con MMR (Maximal Marginal Relevance).
        Input: pregunta o frase especifica sobre derecho de insolvencia colombiano.
        Returns: hasta 6 pasajes relevantes con fuente y pagina.
        """
        docs = vector_store.max_marginal_relevance_search(
            query, k=6, fetch_k=18, lambda_mult=0.5
        )
        if not docs:
            return "No se encontraron documentos relevantes."
        passages = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "desconocido")
            pg = doc.metadata.get("page", "?")
            text = doc.page_content[:600]
            passages.append(f"[{i}] (Fuente: {src}, p.{pg})\n{text}")
        return "\n\n".join(passages)

    return vector_search


def build_all_tools(vector_store: VectorStore):
    """Retorna la lista completa de herramientas para el agente."""
    from src.agent.web_fallback import web_search_tool

    return [build_vector_search_tool(vector_store)] + KG_TOOLS + [web_search_tool]
