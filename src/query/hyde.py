"""
src/query/hyde.py

Hypothetical Document Embeddings (HyDE).
Dada una consulta del usuario, genera un documento hipotetico que responderia
a esa pregunta, y luego usa ese documento para buscar chunks mas alineados
semanticamente en el vector store.

Util para preguntas cortas o ambiguas donde el embedding directo de la
pregunta no captura bien la intencion.
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un experto en derecho de insolvencia empresarial colombiano. "
     "Cuando recibas una pregunta, escribe un fragmento de texto legal "
     "(100-150 palabras) que RESPONDERÍA esa pregunta, como si fuera "
     "extraído de un documento jurídico real sobre la Ley 1116 de 2006, "
     "el Decreto 806 de 2020, o normativa relacionada. "
     "No respondas directamente la pregunta; genera únicamente el texto "
     "hipotético del documento."),
    ("human", "{question}"),
])


def build_hyde_chain(llm: BaseChatModel):
    """
    Retorna un chain: question -> texto_documento_hipotetico.
    Uso: texto_hipotetico = chain.invoke({"question": consulta_usuario})
    """
    return _HYDE_PROMPT | llm | StrOutputParser()


def apply_hyde(llm: BaseChatModel, question: str) -> str:
    """
    Wrapper de conveniencia.
    Genera y retorna el texto del documento hipotetico para la pregunta dada.
    """
    chain = build_hyde_chain(llm)
    return chain.invoke({"question": question})
