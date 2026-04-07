"""
tests/test_metrics.py

Tests de las metricas de evaluacion de retrieval (Fase 6).
No requieren API key.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import (
    recall_at_k, precision_at_k, reciprocal_rank,
    mean_reciprocal_rank, ndcg_at_k, evaluate_retrieval
)


class TestRecall:
    def test_perfect_recall(self):
        assert recall_at_k({"A", "B"}, ["A", "B", "C"], 3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k({"A", "B", "C", "D"}, ["A", "X", "B"], 5) == 0.5

    def test_zero_recall(self):
        assert recall_at_k({"Z"}, ["A", "B", "C"], 3) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(set(), ["A", "B"], 2) == 0.0


class TestPrecision:
    def test_perfect_precision(self):
        assert precision_at_k({"A", "B"}, ["A", "B", "C"], 2) == 1.0

    def test_half_precision(self):
        assert precision_at_k({"A"}, ["A", "B"], 2) == 0.5

    def test_zero_k(self):
        assert precision_at_k({"A"}, ["A"], 0) == 0.0


class TestMRR:
    def test_first_position(self):
        assert reciprocal_rank({"A"}, ["A", "B", "C"]) == 1.0

    def test_second_position(self):
        assert reciprocal_rank({"B"}, ["A", "B", "C"]) == 0.5

    def test_not_found(self):
        assert reciprocal_rank({"D"}, ["A", "B", "C"]) == 0.0

    def test_mean_mrr(self):
        queries = [
            {"relevant": {"A"}, "retrieved": ["A", "B"]},  # rr=1.0
            {"relevant": {"B"}, "retrieved": ["A", "B"]},  # rr=0.5
        ]
        assert abs(mean_reciprocal_rank(queries) - 0.75) < 1e-6


class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k({"A", "B"}, ["A", "B", "C"], 3) == 1.0

    def test_imperfect_ranking(self):
        val = ndcg_at_k({"A", "B"}, ["X", "A", "B"], 3)
        assert 0 < val < 1.0

    def test_empty_relevant(self):
        assert ndcg_at_k(set(), ["A", "B"], 2) == 0.0


class TestEvaluateRetrieval:
    def test_returns_all_metrics(self):
        result = evaluate_retrieval({"A"}, ["A", "B", "C"], k_values=[3])
        assert "recall@3" in result
        assert "precision@3" in result
        assert "ndcg@3" in result
        assert "mrr" in result
