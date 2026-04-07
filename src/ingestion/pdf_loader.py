"""
src/ingestion/pdf_loader.py

Carga todos los PDFs del corpus desde DOCS_DIR, extrae texto via PyMuPDF (fitz),
y adjunta metadata (nombre de archivo, pagina, referencias a normas citadas).
"""
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from src.config import DOCS_DIR


# Patrones regex para detectar citas de normas en el texto (usados como tags de metadata)
_NORM_PATTERNS = {
    "Ley_1116_2006": re.compile(r"[Ll]ey\s+1116\s+de\s+2006"),
    "Decreto_806_2020": re.compile(r"[Dd]ecreto\s+806\s+de\s+2020"),
    "Decreto_772_2020": re.compile(r"[Dd]ecreto\s+772\s+de\s+2020"),
    "Ley_2445_2025": re.compile(r"[Ll]ey\s+2445\s+de\s+2025"),
    "Ley_1564_2012": re.compile(r"[Ll]ey\s+1564\s+de\s+2012"),
    "Ley_2437_2024": re.compile(r"[Ll]ey\s+2437\s+de\s+2024"),
}


def detect_norms(text: str) -> List[str]:
    """Retorna lista de claves de normas encontradas en el texto."""
    return [k for k, pattern in _NORM_PATTERNS.items() if pattern.search(text)]


def load_pdf(path: Path) -> List[Document]:
    """
    Carga un unico archivo PDF.

    Retorna un Document por pagina con metadata:
      - source: str (nombre del archivo)
      - page: int (1-indexed)
      - total_pages: int
      - norms_cited: List[str] (normas mencionadas en la pagina)
    """
    docs: List[Document] = []
    try:
        with fitz.open(str(path)) as pdf:
            total = len(pdf)
            for i, page in enumerate(pdf):
                text = page.get_text("text").strip()
                if not text:
                    continue
                norms = detect_norms(text)
                metadata: Dict[str, Any] = {
                    "source": path.name,
                    "page": i + 1,
                    "total_pages": total,
                    "norms_cited": ",".join(norms) if norms else "none",
                }
                docs.append(Document(page_content=text, metadata=metadata))
    except Exception as exc:
        print(f"[WARN] No se pudo cargar {path.name}: {exc}")
    return docs


def load_corpus(docs_dir: Path = DOCS_DIR) -> List[Document]:
    """
    Carga todos los PDFs en docs_dir.
    Lanza FileNotFoundError si el directorio no existe.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Directorio de documentos no encontrado: {docs_dir}")
    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    all_docs: List[Document] = []
    for path in pdf_paths:
        all_docs.extend(load_pdf(path))
    print(f"[INFO] Cargadas {len(all_docs)} paginas de {len(pdf_paths)} PDFs.")
    return all_docs
