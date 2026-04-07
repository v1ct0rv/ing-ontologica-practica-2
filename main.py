"""
main.py

Punto de entrada principal del sistema Knowledge Graph RAG.
Ejecuta el agente con una pregunta desde la linea de comandos
o inicia el modo interactivo.

Uso:
  python main.py "¿Cuáles son los requisitos para la reorganización?"
  python main.py --interactive
  python main.py --ingest   (ejecutar pipeline de ingesta una vez)
"""
import sys
import os
import warnings
import logging

# Silenciar warnings no criticos
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*renamed to.*ddgs.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Silenciar el LOAD REPORT de sentence-transformers (usa print, no logging)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Monkey-patch para silenciar el LOAD REPORT de safetensors
import builtins
_original_print = builtins.print
def _filtered_print(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    if any(skip in text for skip in ["LOAD REPORT", "UNEXPECTED", "Loading weights", "unauthenticated"]):
        return
    _original_print(*args, **kwargs)
builtins.print = _filtered_print

sys.path.insert(0, os.path.dirname(__file__))


def setup():
    """Configura LangSmith y variables de entorno."""
    from src.tracing.langsmith_setup import configure_langsmith, is_tracing_enabled
    configure_langsmith()
    if is_tracing_enabled():
        print("[INFO] LangSmith tracing activo")
    else:
        print("[WARN] LangSmith no configurado (LANGCHAIN_API_KEY no definida)")


def run_ingest():
    """Ejecuta el pipeline de ingesta: PDF -> chunks -> ChromaDB."""
    print("=" * 60)
    print("PIPELINE DE INGESTA")
    print("=" * 60)
    from src.ingestion.vector_store import run_ingestion_pipeline
    store = run_ingestion_pipeline()
    count = store._collection.count()
    print(f"\n[OK] Pipeline completado. {count} chunks en ChromaDB.")


def run_single_query(question: str):
    """Ejecuta una sola pregunta contra el agente."""
    setup()
    from src.agent.graph import run_query

    print("=" * 60)
    print(f"PREGUNTA: {question}")
    print("=" * 60)

    result = run_query(question)

    print(f"\nESTRATEGIA: {result.get('route', 'N/A')}")
    print(f"REINTENTOS: {result.get('retry_count', 0)}")
    print(f"WEB FALLBACK: {'Sí' if result.get('used_web_fallback') else 'No'}")
    print(f"\nRESPUESTA:")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)

    if result.get("sources"):
        print(f"\nFUENTES ({len(result['sources'])}):")
        for s in result["sources"][:10]:
            print(f"  - {s}")

    if result.get("critique"):
        print(f"\nCRITICA: {result['critique'][:200]}")

    return result


def run_interactive():
    """Modo interactivo: pregunta-respuesta en bucle."""
    setup()
    from src.agent.graph import run_query

    print("=" * 60)
    print("KNOWLEDGE GRAPH RAG — Insolvencia Empresarial Colombia")
    print("Escriba 'salir' para terminar")
    print("=" * 60)

    while True:
        try:
            question = input("\n> Pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        if not question or question.lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        result = run_query(question)

        print(f"\n[Estrategia: {result.get('route', 'N/A')} | "
              f"Reintentos: {result.get('retry_count', 0)} | "
              f"Web: {'Sí' if result.get('used_web_fallback') else 'No'}]")
        print()
        print(result["answer"])

        if result.get("sources"):
            print(f"\nFuentes: {', '.join(result['sources'][:5])}")


def run_evaluation():
    """Ejecuta evaluacion con metricas de retrieval sobre ejemplos."""
    setup()
    from src.evaluation.metrics import evaluate_retrieval
    from src.ingestion.vector_store import load_vector_store

    print("=" * 60)
    print("EVALUACION DEL SISTEMA RAG")
    print("=" * 60)

    # Queries de evaluacion con documentos relevantes conocidos
    eval_queries = [
        {
            "question": "¿Qué es la cesación de pagos según la Ley 1116?",
            "relevant_docs": {"ley-1116-del-27-de-diciembre-de-2006.pdf",
                              "ley insolvencia 2006.pdf"},
        },
        {
            "question": "¿Cuáles son los requisitos para la reorganización empresarial?",
            "relevant_docs": {"Cartilla_Ley_1116_ 2006.pdf",
                              "ley-1116-del-27-de-diciembre-de-2006.pdf"},
        },
        {
            "question": "¿Qué establece el Decreto 806 de 2020?",
            "relevant_docs": {"DECRETO 772 DEL 3 DE JUNIO DE 2020.pdf"},
        },
    ]

    store = load_vector_store()

    for eq in eval_queries:
        docs = store.max_marginal_relevance_search(eq["question"], k=10, fetch_k=30)
        retrieved = [d.metadata.get("source", "") for d in docs]
        metrics = evaluate_retrieval(eq["relevant_docs"], retrieved, k_values=[5, 10])

        print(f"\nQuery: {eq['question'][:60]}...")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python main.py 'tu pregunta aquí'")
        print("  python main.py --interactive")
        print("  python main.py --ingest")
        print("  python main.py --evaluate")
        sys.exit(0)

    arg = sys.argv[1]

    if arg == "--ingest":
        run_ingest()
    elif arg == "--interactive":
        run_interactive()
    elif arg == "--evaluate":
        run_evaluation()
    else:
        run_single_query(arg)
