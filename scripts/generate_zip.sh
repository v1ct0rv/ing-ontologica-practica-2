#!/bin/bash
# Genera el ZIP de entrega: practica2-equipo-08.zip
# Según los requerimientos de Practica2.md:
#   - Informe técnico (PDF)
#   - Código fuente (Python, organizado y documentado)
#   - Carpeta con documentos (50 PDFs) y ontología .ttl
#
# Uso: bash scripts/generate_zip.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ZIP_NAME="practica2-equipo-08.zip"
OUTPUT_PATH="$PROJECT_DIR/$ZIP_NAME"

cd "$PROJECT_DIR"

# 1. Generar PDF si no existe o si el .md es más reciente
if [ ! -f report/technical_report.pdf ] || [ report/technical_report.md -nt report/technical_report.pdf ]; then
  echo "[1/3] Generando PDF del informe..."
  bash scripts/generate_pdf.sh
else
  echo "[1/3] PDF ya está actualizado."
fi

# 2. Limpiar artefactos antes de empaquetar
echo "[2/3] Preparando archivos..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 3. Crear ZIP
echo "[3/3] Creando $ZIP_NAME..."
rm -f "$OUTPUT_PATH"

zip -r "$OUTPUT_PATH" \
  main.py \
  requirements.txt \
  .env.example \
  README.md \
  \
  src/__init__.py \
  src/config.py \
  src/embeddings.py \
  src/ingestion/__init__.py \
  src/ingestion/pdf_loader.py \
  src/ingestion/semantic_chunker.py \
  src/ingestion/vector_store.py \
  src/kg/__init__.py \
  src/kg/ontology_manager.py \
  src/kg/graphdb_client.py \
  src/kg/sparql_tools.py \
  src/query/__init__.py \
  src/query/hyde.py \
  src/query/decomposer.py \
  src/query/router.py \
  src/agent/__init__.py \
  src/agent/state.py \
  src/agent/tools.py \
  src/agent/nodes.py \
  src/agent/graph.py \
  src/agent/web_fallback.py \
  src/evaluation/__init__.py \
  src/evaluation/metrics.py \
  src/evaluation/llm_judge.py \
  src/tracing/__init__.py \
  src/tracing/langsmith_setup.py \
  \
  tests/test_metrics.py \
  tests/test_query_transform.py \
  \
  notebooks/demo.ipynb \
  \
  ontology/insolvencia.ttl \
  ontology/graphdb_repo_config.ttl \
  ontology/sparql/select_queries.sparql \
  ontology/sparql/filter_queries.sparql \
  ontology/sparql/update_queries.sparql \
  ontology/sparql/inference_cases.sparql \
  \
  content/docs/*.pdf \
  \
  report/technical_report.pdf \
  report/technical_report.md \
  report/screenshots/*.png \
  \
  -x "*.DS_Store" "*__pycache__*" "*.pyc"

echo ""
echo "=========================================="
echo "ZIP generado: $ZIP_NAME"
echo "Tamaño: $(du -h "$OUTPUT_PATH" | cut -f1)"
echo "=========================================="
echo ""
echo "Contenido:"
zipinfo -1 "$OUTPUT_PATH" | head -20
echo "... ($(zipinfo -1 "$OUTPUT_PATH" | wc -l | tr -d ' ') archivos total)"
