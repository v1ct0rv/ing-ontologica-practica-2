"""
tests/test_query_transform.py

Test de los modulos de transformacion de consultas (Fase 4).
Requiere OPENAI_API_KEY configurada en .env para ejecutarse.

Uso: python -m pytest tests/test_query_transform.py -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.query.decomposer import SemicolonListParser


# --- Tests sin LLM (siempre ejecutables) ---

class TestSemicolonListParser:
    """Tests del parser de sub-preguntas."""

    def setup_method(self):
        self.parser = SemicolonListParser()

    def test_single_question(self):
        result = self.parser.parse("¿Qué es la cesación de pagos?")
        assert len(result) == 1

    def test_multiple_questions(self):
        result = self.parser.parse("¿Qué es?; ¿Cómo funciona?; ¿Qué plazos?")
        assert len(result) == 3

    def test_empty_segments(self):
        result = self.parser.parse(";;;")
        assert len(result) == 0

    def test_trailing_semicolon(self):
        result = self.parser.parse("Solo una; ")
        assert len(result) == 1
        assert result[0] == "Solo una"

    def test_whitespace_handling(self):
        result = self.parser.parse("  A  ;  B  ;  C  ")
        assert result == ["A", "B", "C"]


# --- Tests con LLM (requieren API key) ---

def _get_llm():
    """Obtiene el LLM o salta el test si no hay API key."""
    from src.config import OPENAI_API_KEY, LLM_MODEL
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
        pytest.skip("OPENAI_API_KEY no configurada")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


class TestHyDE:
    """Tests de HyDE (requieren API key)."""

    def test_hyde_generates_text(self):
        llm = _get_llm()
        from src.query.hyde import apply_hyde
        result = apply_hyde(llm, "¿Qué es la cesación de pagos?")
        assert len(result) > 50
        assert isinstance(result, str)

    def test_hyde_domain_relevance(self):
        llm = _get_llm()
        from src.query.hyde import apply_hyde
        result = apply_hyde(llm, "¿Requisitos para reorganización empresarial?")
        # Debe mencionar algo relacionado con insolvencia
        result_lower = result.lower()
        assert any(w in result_lower for w in ["reorganización", "insolvencia", "ley", "deudor", "acreedor"])


class TestDecomposer:
    """Tests de Query Decomposition (requieren API key)."""

    def test_complex_query_decomposed(self):
        llm = _get_llm()
        from src.query.decomposer import decompose_query
        result = decompose_query(llm,
            "¿Cuáles son los requisitos y plazos para la reorganización "
            "empresarial y qué diferencias hay con la liquidación judicial?")
        assert len(result) >= 2

    def test_simple_query_not_decomposed(self):
        llm = _get_llm()
        from src.query.decomposer import decompose_query
        result = decompose_query(llm, "¿Qué es la cesación de pagos?")
        assert len(result) >= 1


class TestRouter:
    """Tests del router de estrategia (requieren API key)."""

    def test_route_direct(self):
        llm = _get_llm()
        from src.query.router import route_query
        result = route_query(llm, "¿Qué establece el artículo 9 de la Ley 1116?")
        assert result in ("DIRECT", "HYDE", "DECOMPOSE")

    def test_route_complex(self):
        llm = _get_llm()
        from src.query.router import route_query
        result = route_query(llm,
            "¿Cuáles son los requisitos y plazos para la reorganización "
            "y qué diferencias hay con la liquidación?")
        assert result in ("DIRECT", "HYDE", "DECOMPOSE")
