"""
src/evaluation/metrics.py

Implementa metricas de evaluacion para el sistema RAG:
  - Retrieval: Recall@k, Precision@k, MRR, nDCG
  - Answer: RAGAS (faithfulness, answer_relevancy)

Las metricas de retrieval se calculan comparando documentos recuperados
contra un conjunto de documentos relevantes anotados manualmente (ground truth).
"""
import math
from typing import List, Set, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Metricas de Retrieval
# ---------------------------------------------------------------------------

def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """
    Recall@k = |relevant ∩ retrieved[:k]| / |relevant|

    Mide que fraccion de los documentos relevantes fueron recuperados
    en los primeros k resultados.
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(relevant & top_k) / len(relevant)


def precision_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """
    Precision@k = |relevant ∩ retrieved[:k]| / k

    Mide que fraccion de los primeros k resultados son relevantes.
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / k


def reciprocal_rank(relevant: Set[str], retrieved: List[str]) -> float:
    """
    Contribucion MRR para una sola query: 1/rank del primer resultado relevante.
    Retorna 0.0 si ningun resultado es relevante.
    """
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(queries_results: List[Dict[str, Any]]) -> float:
    """
    Calcula Mean Reciprocal Rank sobre multiples queries.

    queries_results: lista de {"relevant": set, "retrieved": list}
    """
    rr_scores = [
        reciprocal_rank(q["relevant"], q["retrieved"])
        for q in queries_results
    ]
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def dcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Discounted Cumulative Gain @k."""
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """
    Normalised DCG@k.

    Mide la calidad del ranking, penalizando resultados relevantes
    que aparecen en posiciones bajas.
    """
    actual_dcg = dcg_at_k(relevant, retrieved, k)
    # Ideal: todos los relevantes primero
    ideal_retrieved = list(relevant)[:k] + [
        f"__pad_{i}" for i in range(max(0, k - len(relevant)))
    ]
    ideal_dcg = dcg_at_k(relevant, ideal_retrieved, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluacion consolidada de retrieval
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    relevant: Set[str],
    retrieved: List[str],
    k_values: List[int] = [5, 10],
) -> Dict[str, float]:
    """
    Calcula todas las metricas de retrieval para una query.

    Retorna dict con:
      recall@5, recall@10, precision@5, precision@10, mrr, ndcg@5, ndcg@10
    """
    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(relevant, retrieved, k)
        results[f"precision@{k}"] = precision_at_k(relevant, retrieved, k)
        results[f"ndcg@{k}"] = ndcg_at_k(relevant, retrieved, k)
    results["mrr"] = reciprocal_rank(relevant, retrieved)
    return results


# ---------------------------------------------------------------------------
# Metricas de respuesta con RAGAS (wrapper)
# ---------------------------------------------------------------------------

def compute_ragas_metrics(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dict[str, float]:
    """
    Calcula metricas RAGAS: answer_relevancy y faithfulness.

    Requiere: pip install ragas datasets
    Retorna dict con nombres de metricas como claves.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[answer_relevancy, faithfulness])
        return dict(result)
    except ImportError:
        return {"error": "RAGAS no instalado. Ejecute: pip install ragas datasets"}
    except Exception as exc:
        return {"error": f"Error RAGAS: {exc}"}
