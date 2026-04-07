"""
src/query/router.py

Clasifica la consulta del usuario para decidir la estrategia de recuperacion:
  "DIRECT"     — busqueda vectorial directa (preguntas factuales, definiciones)
  "HYDE"       — generar documento hipotetico (preguntas cortas/ambiguas)
  "DECOMPOSE"  — descomponer en sub-preguntas (preguntas complejas multi-hop)

El router es el primer nodo del flujo LangGraph y determina como se
transformara la consulta antes de pasar al agente ReAct.
"""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


QueryStrategy = Literal["DIRECT", "HYDE", "DECOMPOSE"]

_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Clasifica la siguiente pregunta sobre derecho de insolvencia empresarial "
     "colombiano en una de estas tres estrategias de búsqueda. "
     "Responde ÚNICAMENTE con una de estas palabras: DIRECT, HYDE, DECOMPOSE.\n\n"
     "- DIRECT: pregunta clara y específica sobre un concepto, artículo o "
     "procedimiento. Ejemplo: '¿Qué establece el artículo 9 de la Ley 1116?'\n"
     "- HYDE: pregunta corta, vaga o ambigua donde generar un documento "
     "hipotético mejoraría la búsqueda. Ejemplo: '¿Qué pasa cuando no se paga?'\n"
     "- DECOMPOSE: pregunta compleja que contiene múltiples sub-preguntas o "
     "condicionales. Ejemplo: '¿Cuáles son los requisitos y plazos para la "
     "reorganización y qué diferencias hay con la liquidación?'"),
    ("human", "{question}"),
])


def build_router(llm: BaseChatModel):
    """Retorna chain: question -> strategy_string."""
    return _ROUTER_PROMPT | llm | StrOutputParser()


def route_query(llm: BaseChatModel, question: str) -> QueryStrategy:
    """
    Clasifica la pregunta y retorna la estrategia de transformacion.
    Si la clasificacion falla, retorna 'DIRECT' como default seguro.
    """
    try:
        chain = build_router(llm)
        raw = chain.invoke({"question": question}).strip().upper()
        if raw in ("DIRECT", "HYDE", "DECOMPOSE"):
            return raw  # type: ignore
        return "DIRECT"
    except Exception:
        return "DIRECT"
