#!/bin/bash
# Genera el informe técnico en PDF desde el markdown
# Uso: bash scripts/generate_pdf.sh

cd "$(dirname "$0")/../report"

echo "Generando PDF del informe técnico..."

pandoc technical_report.md \
  -o technical_report.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=2.5cm \
  -V fontsize=11pt \
  -V lang=es \
  -V mainfont="Helvetica" \
  -V monofont="Menlo" \
  --toc \
  --toc-depth=3

if [ $? -eq 0 ]; then
  echo "PDF generado: report/technical_report.pdf ($(du -h technical_report.pdf | cut -f1))"
else
  echo "Error generando PDF. Verifica que pandoc y xelatex estén instalados."
  echo "  brew install pandoc"
  echo "  (xelatex viene con MacTeX o BasicTeX)"
  exit 1
fi
