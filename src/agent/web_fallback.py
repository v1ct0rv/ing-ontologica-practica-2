"""
src/agent/web_fallback.py

Busqueda web via DuckDuckGo como fallback cuando la recuperacion
del vector store + KG no produce una respuesta aprobada despues
de MAX_RETRIES intentos.
"""
from langchain_core.tools import tool
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5) -> str:
    """
    Busca en DuckDuckGo informacion sobre insolvencia empresarial colombiana.
    Retorna string formateado con snippets de resultados.
    """
    search_query = f"insolvencia empresarial Colombia Ley 1116 2006 {query}"
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=max_results):
                results.append(
                    f"[{r['title']}]\n{r['body']}\nURL: {r['href']}"
                )
    except Exception as exc:
        return f"Error en busqueda web: {exc}"
    return "\n\n".join(results) if results else "No se encontraron resultados web."


@tool
def web_search_tool(query: str) -> str:
    """
    Busqueda web de respaldo para preguntas sobre insolvencia empresarial colombiana.
    Usar SOLO cuando el vector store y el Knowledge Graph son insuficientes.
    Input: la pregunta del usuario o una consulta de busqueda refinada.
    """
    return search_web(query)
