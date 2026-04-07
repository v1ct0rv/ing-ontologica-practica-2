"""
src/evaluation/llm_judge.py

Usa un LLM como juez para puntuar respuestas generadas en una escala
de 1 a 5 en tres dimensiones:
  - Relevancia: ¿La respuesta aborda la pregunta?
  - Fidelidad: ¿La respuesta se basa en el contexto sin alucinar?
  - Precision Legal: ¿Cita correctamente normas y articulos?

Requerido por la practica: "Incluir evaluacion con LLM como juez."
"""
from typing import Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un juez experto en derecho colombiano de insolvencia empresarial. "
     "Tu tarea es evaluar la calidad de una respuesta generada por un sistema RAG.\n\n"
     "Evalúa la respuesta en una escala de 1 a 5 en TRES dimensiones:\n"
     "- Relevancia (1-5): ¿La respuesta aborda completamente la pregunta?\n"
     "- Fidelidad (1-5): ¿La respuesta se basa en el contexto sin inventar hechos?\n"
     "- Precision_Legal (1-5): ¿Cita correctamente las normas, artículos y procedimientos?\n\n"
     "Responde EXACTAMENTE en este formato (sin texto adicional antes):\n"
     "Relevancia: <1-5>\n"
     "Fidelidad: <1-5>\n"
     "Precision_Legal: <1-5>\n"
     "Justificacion: <una o dos frases explicando la evaluación>"),
    ("human",
     "PREGUNTA: {question}\n\n"
     "RESPUESTA A EVALUAR:\n{answer}\n\n"
     "CONTEXTO FUENTE:\n{context}"),
])


def llm_judge_score(
    llm: BaseChatModel,
    question: str,
    answer: str,
    context: str,
) -> Dict[str, Any]:
    """
    Evalua una respuesta usando un LLM como juez.

    Retorna dict con:
      - relevancia: float (1-5)
      - fidelidad: float (1-5)
      - precision_legal: float (1-5)
      - justificacion: str
      - promedio: float (promedio de las 3 dimensiones)
    """
    chain = _JUDGE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "question": question,
        "answer": answer,
        "context": context[:2000],
    })

    scores: Dict[str, Any] = {}
    for line in raw.strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            try:
                scores[key] = float(val)
            except ValueError:
                scores[key] = val  # para justificacion (texto)

    # Calcular promedio de las dimensiones numericas
    numeric = [v for v in scores.values() if isinstance(v, (int, float))]
    scores["promedio"] = sum(numeric) / len(numeric) if numeric else 0.0
    return scores


def evaluate_batch(
    llm: BaseChatModel,
    test_cases: list,
) -> list:
    """
    Evalua un lote de casos de prueba.

    test_cases: lista de dicts con keys "question", "answer", "context"
    Retorna lista de dicts con scores por cada caso.
    """
    results = []
    for case in test_cases:
        score = llm_judge_score(
            llm=llm,
            question=case["question"],
            answer=case["answer"],
            context=case.get("context", ""),
        )
        score["question"] = case["question"]
        results.append(score)
    return results
