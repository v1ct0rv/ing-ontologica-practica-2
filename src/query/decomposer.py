"""
src/query/decomposer.py

Descompone una pregunta legal compleja de multi-hop en 2-4 sub-preguntas
mas simples y especificas. Cada sub-pregunta se recupera independientemente
y los resultados se fusionan antes de la generacion de respuesta.

Util cuando la consulta del usuario contiene multiples preguntas o
condicionales (ej. "¿Qué es la cesación de pagos y cuáles son sus
consecuencias legales bajo la Ley 1116?").
"""
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser


_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un experto en análisis de preguntas legales sobre insolvencia "
     "empresarial colombiana. Descompón la siguiente pregunta compleja en "
     "2-4 sub-preguntas simples y específicas que juntas cubran toda la "
     "información solicitada.\n\n"
     "Reglas:\n"
     "- Devuelve ÚNICAMENTE las sub-preguntas separadas por punto y coma (;)\n"
     "- No incluyas numeración, viñetas ni texto adicional\n"
     "- Cada sub-pregunta debe ser auto-contenida y comprensible por sí sola\n"
     "- Si la pregunta ya es simple, devuélvela tal cual"),
    ("human", "{question}"),
])


class SemicolonListParser(BaseOutputParser[List[str]]):
    """Parser que separa por punto y coma en lugar de coma,
    ya que el texto legal en español usa muchas comas internamente."""

    def parse(self, text: str) -> List[str]:
        return [q.strip() for q in text.split(";") if q.strip()]

    @property
    def _type(self) -> str:
        return "semicolon_list"


def build_decomposer(llm: BaseChatModel):
    """Retorna chain: question -> List[str] de sub-preguntas."""
    return _DECOMPOSE_PROMPT | llm | SemicolonListParser()


def decompose_query(llm: BaseChatModel, question: str) -> List[str]:
    """
    Retorna lista de sub-preguntas derivadas de la pregunta original.
    Si la descomposicion falla o retorna solo 1 item, devuelve [question].
    """
    try:
        chain = build_decomposer(llm)
        parts = chain.invoke({"question": question})
        return parts if len(parts) > 1 else [question]
    except Exception:
        return [question]
