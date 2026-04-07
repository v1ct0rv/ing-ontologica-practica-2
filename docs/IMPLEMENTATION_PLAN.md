# Implementation Plan: Knowledge Graph RAG System
## Colombian Business Insolvency Law (Ley 1116 de 2006)
### Ingenieria Ontologica — Practica 2

**Due date:** April 10, 2026
**Start date:** March 28, 2026
**Total time budget:** 13 days

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Phase 1: Project Setup & Ontology (Days 1–3)](#phase-1-project-setup--ontology-days-13)
3. [Phase 2: Document Ingestion & Vector Store (Days 3–5)](#phase-2-document-ingestion--vector-store-days-35)
4. [Phase 3: Knowledge Graph Integration (Days 5–7)](#phase-3-knowledge-graph-integration-days-57)
5. [Phase 4: Query Transformation (Days 7–8)](#phase-4-query-transformation-days-78)
6. [Phase 5: Agent Architecture (Days 8–10)](#phase-5-agent-architecture-days-810)
7. [Phase 6: LangSmith & Evaluation (Days 10–11)](#phase-6-langsmith--evaluation-days-1011)
8. [Phase 7: Documentation & Delivery (Days 11–13)](#phase-7-documentation--delivery-days-1113)
9. [Dependency Map](#dependency-map)
10. [Risk Register](#risk-register)

---

## Repository Structure

Create the following tree before writing any code. Every file listed here is
referenced by at least one phase below.

```
Practica 2/
├── content/
│   └── docs/                        # 50 PDFs — already present
├── docs/
│   └── IMPLEMENTATION_PLAN.md       # this file
├── agents/                          # agent markdown specs (already present)
├── ontology/
│   ├── insolvencia.ttl              # OWL ontology (Turtle)
│   └── sparql/
│       ├── select_queries.sparql
│       ├── filter_queries.sparql
│       ├── update_queries.sparql
│       └── inference_cases.sparql
├── src/
│   ├── __init__.py
│   ├── config.py                    # env vars, paths, constants
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py
│   │   ├── semantic_chunker.py
│   │   └── vector_store.py
│   ├── kg/
│   │   ├── __init__.py
│   │   ├── ontology_manager.py
│   │   ├── graphdb_client.py
│   │   └── sparql_tools.py
│   ├── query/
│   │   ├── __init__.py
│   │   ├── hyde.py
│   │   ├── decomposer.py
│   │   └── router.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── tools.py
│   │   ├── nodes.py
│   │   ├── graph.py
│   │   └── web_fallback.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── llm_judge.py
│   └── tracing/
│       ├── __init__.py
│       └── langsmith_setup.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_kg.py
│   ├── test_agent.py
│   └── test_metrics.py
├── notebooks/
│   └── demo.ipynb
├── report/
│   └── technical_report.md         # source for PDF export
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Phase 1: Project Setup & Ontology (Days 1–3)

### Step 1.1 — Environment and dependencies

**Files to create:** `requirements.txt`, `pyproject.toml`, `.env.example`, `src/config.py`

**Effort:** 2 hours | **Risk:** Low

#### `requirements.txt`

```text
# Core LLM / agent
langchain>=0.3
langchain-community>=0.3
langchain-openai>=0.2
langchain-chroma>=0.1
langgraph>=0.2
langsmith>=0.1

# Vector store & embeddings
chromadb>=0.5
faiss-cpu>=1.8          # fallback if Chroma has issues
sentence-transformers>=3.0

# PDF ingestion
pymupdf>=1.24           # fitz — fastest PDF parser
pypdf>=4.0              # fallback / metadata
unstructured[pdf]>=0.14

# Knowledge graph
rdflib>=7.0
sparqlwrapper>=2.0

# Web fallback
duckduckgo-search>=6.0
requests>=2.31
beautifulsoup4>=4.12

# Evaluation
ragas>=0.1.18
numpy>=1.26
pandas>=2.2
scikit-learn>=1.5

# Tracing
python-dotenv>=1.0
tqdm>=4.66
loguru>=0.7
```

#### `src/config.py`

```python
"""
Central configuration — reads .env and exposes typed constants.
All path and credential references in the codebase must import from here.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- Project root ----------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# --- Document corpus -------------------------------------------------------
DOCS_DIR: Path = ROOT / "content" / "docs"

# --- Vector store ----------------------------------------------------------
CHROMA_PERSIST_DIR: Path = ROOT / ".chroma_db"
CHROMA_COLLECTION_NAME: str = "insolvencia_chunks"
EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_BREAKPOINT_TYPE: str = "percentile"   # for SemanticChunker
CHUNK_BREAKPOINT_THRESHOLD: float = 95.0

# --- Ontology / KG ---------------------------------------------------------
ONTOLOGY_PATH: Path = ROOT / "ontology" / "insolvencia.ttl"
GRAPHDB_ENDPOINT: str = os.getenv("GRAPHDB_ENDPOINT", "http://localhost:7200")
GRAPHDB_REPOSITORY: str = os.getenv("GRAPHDB_REPOSITORY", "insolvencia")
GRAPHDB_SPARQL_ENDPOINT: str = (
    f"{GRAPHDB_ENDPOINT}/repositories/{GRAPHDB_REPOSITORY}"
)
GRAPHDB_UPDATE_ENDPOINT: str = (
    f"{GRAPHDB_ENDPOINT}/repositories/{GRAPHDB_REPOSITORY}/statements"
)
ONTOLOGY_BASE_URI: str = "http://www.unal.edu.co/ontologies/insolvencia#"

# --- LLM -------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = 0.0

# --- LangSmith -------------------------------------------------------------
LANGCHAIN_TRACING_V2: str = "true"
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "insolvencia-kg-rag")

# --- Agent -----------------------------------------------------------------
MAX_RETRIES: int = 3
RETRIEVAL_K: int = 6       # number of chunks per retrieval call
MMR_LAMBDA: float = 0.5    # diversity vs. relevance balance
```

#### `.env.example`

```dotenv
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=insolvencia-kg-rag
GRAPHDB_ENDPOINT=http://localhost:7200
GRAPHDB_REPOSITORY=insolvencia
LLM_MODEL=gpt-4o-mini
```

**Dependencies on other steps:** None (this is the foundation).

---

### Step 1.2 — OWL Ontology design

**Files to create:** `ontology/insolvencia.ttl`

**Effort:** 6 hours | **Risk:** Medium (domain modeling requires legal precision)

The ontology must satisfy the following OWL requirements from the rubric:

| Requirement | Target |
|---|---|
| Classes | >= 10 |
| DisjointClasses | >= 2 pairs |
| subClassOf | >= 3 |
| subPropertyOf | >= 2 |
| Object/datatype properties with domain/range | >= 10 |
| Named individuals per class | >= 4 |
| inverseOf | >= 2 |
| allValuesFrom | >= 1 |
| someValuesFrom | >= 1 |
| cardinality restriction | >= 1 |
| Logical constructor | union OR intersection OR complement |

#### Class hierarchy (domain analysis)

```
owl:Thing
 └── ins:EntidadJuridica
      ├── ins:Deudor                        # empresa insolvente
      │    ├── ins:PersonaJuridica
      │    └── ins:PersonaNatural
      ├── ins:Acreedor
      │    ├── ins:AcreedorPrivilegiado      # subClassOf Acreedor
      │    └── ins:AcreedorOrdinario         # subClassOf Acreedor
      └── ins:Liquidador
 └── ins:ProcedimientoInsolvencia
      ├── ins:Reorganizacion                # Art. 5 Ley 1116
      └── ins:Liquidacion                  # Art. 47 Ley 1116
 └── ins:ActoJuridico
      ├── ins:AutoAdmision
      └── ins:AcuerdoReorganizacion
 └── ins:Obligacion
 └── ins:Garantia
 └── ins:OrganoDecisorio
      ├── ins:JuntaAcreedores
      └── ins:Superintendencia
 └── ins:NormaLegal
      ├── ins:Ley
      └── ins:Decreto
```

#### Full ontology file (`ontology/insolvencia.ttl`)

```turtle
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix ins:  <http://www.unal.edu.co/ontologies/insolvencia#> .

###############################################################################
# Ontology declaration
###############################################################################
<http://www.unal.edu.co/ontologies/insolvencia>
    a owl:Ontology ;
    rdfs:label "Ontologia de Insolvencia Empresarial Colombia"@es ;
    rdfs:comment "OWL ontology covering Ley 1116 de 2006 and related norms."@es .

###############################################################################
# Classes
###############################################################################
ins:EntidadJuridica   a owl:Class ; rdfs:label "Entidad Juridica"@es .
ins:Deudor            a owl:Class ; rdfs:subClassOf ins:EntidadJuridica ;
                        rdfs:label "Deudor"@es .
ins:PersonaJuridica   a owl:Class ; rdfs:subClassOf ins:Deudor .
ins:PersonaNatural    a owl:Class ; rdfs:subClassOf ins:Deudor .
ins:Acreedor          a owl:Class ; rdfs:subClassOf ins:EntidadJuridica ;
                        rdfs:label "Acreedor"@es .
ins:AcreedorPrivilegiado a owl:Class ; rdfs:subClassOf ins:Acreedor .
ins:AcreedorOrdinario    a owl:Class ; rdfs:subClassOf ins:Acreedor .
ins:Liquidador        a owl:Class ; rdfs:subClassOf ins:EntidadJuridica .
ins:ProcedimientoInsolvencia a owl:Class ; rdfs:label "Procedimiento de Insolvencia"@es .
ins:Reorganizacion    a owl:Class ; rdfs:subClassOf ins:ProcedimientoInsolvencia .
ins:Liquidacion       a owl:Class ; rdfs:subClassOf ins:ProcedimientoInsolvencia .
ins:ActoJuridico      a owl:Class .
ins:AutoAdmision      a owl:Class ; rdfs:subClassOf ins:ActoJuridico .
ins:AcuerdoReorganizacion a owl:Class ; rdfs:subClassOf ins:ActoJuridico .
ins:Obligacion        a owl:Class .
ins:Garantia          a owl:Class .
ins:OrganoDecisorio   a owl:Class .
ins:JuntaAcreedores   a owl:Class ; rdfs:subClassOf ins:OrganoDecisorio .
ins:Superintendencia  a owl:Class ; rdfs:subClassOf ins:OrganoDecisorio .
ins:NormaLegal        a owl:Class .
ins:Ley               a owl:Class ; rdfs:subClassOf ins:NormaLegal .
ins:Decreto           a owl:Class ; rdfs:subClassOf ins:NormaLegal .

# DisjointClasses (>= 2 pairs)
[ a owl:AllDisjointClasses ;
  owl:members ( ins:Reorganizacion ins:Liquidacion ) ] .
[ a owl:AllDisjointClasses ;
  owl:members ( ins:AcreedorPrivilegiado ins:AcreedorOrdinario ) ] .

# Union constructor: un Deudor puede ser PersonaJuridica o PersonaNatural
ins:Deudor owl:equivalentClass [
    a owl:Class ;
    owl:unionOf ( ins:PersonaJuridica ins:PersonaNatural )
] .

###############################################################################
# Object Properties
###############################################################################
ins:tieneAcreedor a owl:ObjectProperty ;
    rdfs:domain ins:Deudor ;
    rdfs:range  ins:Acreedor ;
    rdfs:label  "tiene acreedor"@es .

ins:esAcreedorDe a owl:ObjectProperty ;
    owl:inverseOf ins:tieneAcreedor ;
    rdfs:domain ins:Acreedor ;
    rdfs:range  ins:Deudor .

ins:iniciaEn a owl:ObjectProperty ;
    rdfs:domain ins:Deudor ;
    rdfs:range  ins:ProcedimientoInsolvencia .

ins:esIniciadoPor a owl:ObjectProperty ;
    owl:inverseOf ins:iniciaEn ;
    rdfs:domain ins:ProcedimientoInsolvencia ;
    rdfs:range  ins:Deudor .

ins:emite a owl:ObjectProperty ;
    rdfs:domain ins:OrganoDecisorio ;
    rdfs:range  ins:ActoJuridico .

ins:tieneGarantia a owl:ObjectProperty ;
    rdfs:domain ins:Obligacion ;
    rdfs:range  ins:Garantia .

ins:participaEn a owl:ObjectProperty ;
    rdfs:domain ins:Acreedor ;
    rdfs:range  ins:JuntaAcreedores .

ins:estaReguladoPor a owl:ObjectProperty ;
    rdfs:domain ins:ProcedimientoInsolvencia ;
    rdfs:range  ins:NormaLegal .

ins:tieneObligacion a owl:ObjectProperty ;
    rdfs:domain ins:Deudor ;
    rdfs:range  ins:Obligacion .

ins:designaLiquidador a owl:ObjectProperty ;
    rdfs:domain ins:OrganoDecisorio ;
    rdfs:range  ins:Liquidador .

ins:tieneAcreedorPrivilegiado a owl:ObjectProperty ;
    rdfs:subPropertyOf ins:tieneAcreedor ;   # subPropertyOf (1): privileged creditor IS-A creditor relationship
    rdfs:domain ins:Deudor ;
    rdfs:range  ins:AcreedorPrivilegiado .

ins:esSupervisadoPor a owl:ObjectProperty ;
    rdfs:subPropertyOf ins:estaReguladoPor ; # subPropertyOf (2): supervision is a form of regulation
    rdfs:domain ins:ProcedimientoInsolvencia ;
    rdfs:range  ins:OrganoDecisorio .

###############################################################################
# Datatype Properties
###############################################################################
ins:nombreRazonSocial a owl:DatatypeProperty ;
    rdfs:domain ins:EntidadJuridica ;
    rdfs:range  xsd:string .

ins:nit a owl:DatatypeProperty ;
    rdfs:domain ins:PersonaJuridica ;
    rdfs:range  xsd:string .

ins:fechaAdmision a owl:DatatypeProperty ;
    rdfs:domain ins:ProcedimientoInsolvencia ;
    rdfs:range  xsd:date .

ins:montoObligacion a owl:DatatypeProperty ;
    rdfs:domain ins:Obligacion ;
    rdfs:range  xsd:decimal .

ins:numeroNorma a owl:DatatypeProperty ;
    rdfs:domain ins:NormaLegal ;
    rdfs:range  xsd:string .

ins:anoExpedicion a owl:DatatypeProperty ;
    rdfs:domain ins:NormaLegal ;
    rdfs:range  xsd:gYear .

###############################################################################
# Restrictions
###############################################################################
# allValuesFrom: todos los acreedores de un Deudor deben ser instancias de Acreedor
ins:Deudor rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty ins:tieneAcreedor ;
    owl:allValuesFrom ins:Acreedor
] .

# someValuesFrom: todo Procedimiento de Insolvencia tiene al menos un organo decisorio
ins:ProcedimientoInsolvencia rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty ins:esSupervisadoPor ;
    owl:someValuesFrom ins:OrganoDecisorio
] .

# Cardinality: un Acuerdo de Reorganizacion tiene exactamente 1 Deudor que lo inicia
ins:AcuerdoReorganizacion rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty ins:esIniciadoPor ;
    owl:cardinality 1
] .

###############################################################################
# Named Individuals
###############################################################################
# -- NormaLegal individuals (>= 4)
ins:Ley1116_2006 a ins:Ley ;
    ins:numeroNorma "1116" ;
    ins:anoExpedicion "2006"^^xsd:gYear ;
    rdfs:label "Ley 1116 de 2006"@es .

ins:Decreto806_2020 a ins:Decreto ;
    ins:numeroNorma "806" ;
    ins:anoExpedicion "2020"^^xsd:gYear ;
    rdfs:label "Decreto 806 de 2020"@es .

ins:Decreto772_2020 a ins:Decreto ;
    ins:numeroNorma "772" ;
    ins:anoExpedicion "2020"^^xsd:gYear ;
    rdfs:label "Decreto 772 de 2020"@es .

ins:Ley2445_2025 a ins:Ley ;
    ins:numeroNorma "2445" ;
    ins:anoExpedicion "2025"^^xsd:gYear ;
    rdfs:label "Ley 2445 de 2025"@es .

ins:Ley1564_2012 a ins:Ley ;
    ins:numeroNorma "1564" ;
    ins:anoExpedicion "2012"^^xsd:gYear ;
    rdfs:label "Codigo General del Proceso"@es .

# -- OrganoDecisorio individuals
ins:SuperintendenciaSociedades a ins:Superintendencia ;
    ins:nombreRazonSocial "Superintendencia de Sociedades"^^xsd:string .

ins:SuperintendenciaBancaria a ins:Superintendencia ;
    ins:nombreRazonSocial "Superintendencia Financiera"^^xsd:string .

ins:JuntaEjemplo1 a ins:JuntaAcreedores .
ins:JuntaEjemplo2 a ins:JuntaAcreedores .

# -- ProcedimientoInsolvencia individuals
ins:ReorgEjemplo1 a ins:Reorganizacion ;
    ins:fechaAdmision "2023-03-15"^^xsd:date ;
    ins:estaReguladoPor ins:Ley1116_2006 ;
    ins:esSupervisadoPor ins:SuperintendenciaSociedades .

ins:LiquidEjemplo1 a ins:Liquidacion ;
    ins:fechaAdmision "2022-07-01"^^xsd:date ;
    ins:estaReguladoPor ins:Ley1116_2006 ;
    ins:esSupervisadoPor ins:SuperintendenciaSociedades .

ins:ReorgEjemplo2 a ins:Reorganizacion ;
    ins:estaReguladoPor ins:Decreto806_2020 ;
    ins:esSupervisadoPor ins:SuperintendenciaSociedades .

ins:LiquidEjemplo2 a ins:Liquidacion ;
    ins:estaReguladoPor ins:Decreto772_2020 .

# -- Deudor individuals
ins:EmpresaABC a ins:PersonaJuridica ;
    ins:nombreRazonSocial "Empresa ABC S.A.S."^^xsd:string ;
    ins:nit "900123456-1"^^xsd:string ;
    ins:iniciaEn ins:ReorgEjemplo1 .

ins:EmpresaXYZ a ins:PersonaJuridica ;
    ins:nombreRazonSocial "Corporacion XYZ Ltda."^^xsd:string ;
    ins:nit "800987654-2"^^xsd:string ;
    ins:iniciaEn ins:LiquidEjemplo1 .

ins:JuanPerez a ins:PersonaNatural ;
    ins:nombreRazonSocial "Juan Perez"^^xsd:string ;
    ins:iniciaEn ins:ReorgEjemplo2 .

ins:MariaLopez a ins:PersonaNatural ;
    ins:nombreRazonSocial "Maria Lopez"^^xsd:string .

# -- Acreedor individuals
ins:BancoColombia a ins:AcreedorPrivilegiado ;
    ins:nombreRazonSocial "Bancolombia S.A."^^xsd:string .

ins:BancoPopular a ins:AcreedorPrivilegiado ;
    ins:nombreRazonSocial "Banco Popular S.A."^^xsd:string .

ins:ProveedorA a ins:AcreedorOrdinario ;
    ins:nombreRazonSocial "Proveedor Alfa S.A.S."^^xsd:string .

ins:ProveedorB a ins:AcreedorOrdinario ;
    ins:nombreRazonSocial "Proveedor Beta Ltda."^^xsd:string .

ins:EmpresaABC ins:tieneAcreedor ins:BancoColombia .
ins:EmpresaABC ins:tieneAcreedor ins:ProveedorA .
ins:EmpresaXYZ ins:tieneAcreedor ins:BancoPopular .
ins:EmpresaXYZ ins:tieneAcreedor ins:ProveedorB .

# -- ActoJuridico individuals (>= 4)
ins:Auto_Admision_ReorgEjemplo1 a ins:AutoAdmision ;
    rdfs:label "Auto de admision Reorganizacion Ejemplo 1"@es .
ins:Auto_Admision_LiquidEjemplo1 a ins:AutoAdmision ;
    rdfs:label "Auto de admision Liquidacion Ejemplo 1"@es .
ins:Acuerdo_ReorgEjemplo1 a ins:AcuerdoReorganizacion ;
    rdfs:label "Acuerdo de reorganizacion Ejemplo 1"@es .
ins:Acuerdo_ReorgEjemplo2 a ins:AcuerdoReorganizacion ;
    rdfs:label "Acuerdo de reorganizacion Ejemplo 2"@es .

# -- Obligacion individuals (>= 4)
ins:Obligacion_ABC_Banco a ins:Obligacion ;
    ins:montoObligacion "500000000"^^xsd:decimal ;
    ins:tieneGarantia ins:Garantia_Hipoteca_ABC ;
    rdfs:label "Obligacion Empresa ABC con Bancolombia"@es .
ins:Obligacion_ABC_Proveedor a ins:Obligacion ;
    ins:montoObligacion "150000000"^^xsd:decimal ;
    rdfs:label "Obligacion Empresa ABC con Proveedor Alfa"@es .
ins:Obligacion_XYZ_Banco a ins:Obligacion ;
    ins:montoObligacion "800000000"^^xsd:decimal ;
    ins:tieneGarantia ins:Garantia_Prenda_XYZ ;
    rdfs:label "Obligacion Empresa XYZ con Banco Popular"@es .
ins:Obligacion_XYZ_Proveedor a ins:Obligacion ;
    ins:montoObligacion "200000000"^^xsd:decimal ;
    rdfs:label "Obligacion Empresa XYZ con Proveedor Beta"@es .

# -- Garantia individuals (>= 4)
ins:Garantia_Hipoteca_ABC a ins:Garantia ;
    rdfs:label "Hipoteca sobre inmueble Empresa ABC"@es .
ins:Garantia_Prenda_XYZ a ins:Garantia ;
    rdfs:label "Prenda sobre maquinaria Empresa XYZ"@es .
ins:Garantia_Fianza_JuanPerez a ins:Garantia ;
    rdfs:label "Fianza personal Juan Perez"@es .
ins:Garantia_Pagare_MariaLopez a ins:Garantia ;
    rdfs:label "Pagare firmado por Maria Lopez"@es .
```

**Dependencies:** Step 1.1 (config paths must be set).

---

### Step 1.3 — GraphDB setup and ontology upload

**Files to create:** `src/kg/graphdb_client.py` (partial, completed in Phase 3)

**Effort:** 2 hours | **Risk:** Medium (requires GraphDB server running locally)

1. Download and install GraphDB Free from `https://www.ontotext.com/products/graphdb/`
2. Start GraphDB: `./graphdb -d` (background daemon)
3. Create repository named `insolvencia` via the Workbench UI at `http://localhost:7200`
4. Upload `ontology/insolvencia.ttl` via:
   - GraphDB Workbench: Import > RDF Files > choose `.ttl`
   - OR via curl: `curl -X POST -H "Content-Type: text/turtle" --data-binary @ontology/insolvencia.ttl http://localhost:7200/repositories/insolvencia/statements`

**Verify upload:**
```bash
curl "http://localhost:7200/repositories/insolvencia?query=SELECT+%28COUNT%28*%29+AS+%3Fc%29+WHERE+%7B%3Fs+%3Fp+%3Fo%7D"
```
Expected: count > 80 triples.

**Dependencies:** Step 1.2.

---

## Phase 2: Document Ingestion & Vector Store (Days 3–5)

### Step 2.1 — PDF loading pipeline

**Files to create:** `src/ingestion/pdf_loader.py`

**Effort:** 3 hours | **Risk:** Low

```python
"""
src/ingestion/pdf_loader.py

Loads all PDFs from DOCS_DIR, extracts text via PyMuPDF (fitz),
and attaches metadata (filename, page number, source law references).
"""
import fitz                          # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from src.config import DOCS_DIR


# Regex patterns to detect norm citations in text (used as metadata tags)
_NORM_PATTERNS = {
    "Ley_1116_2006":  re.compile(r"[Ll]ey\s+1116\s+de\s+2006"),
    "Decreto_806_2020": re.compile(r"[Dd]ecreto\s+806\s+de\s+2020"),
    "Decreto_772_2020": re.compile(r"[Dd]ecreto\s+772\s+de\s+2020"),
    "Ley_2445_2025":  re.compile(r"[Ll]ey\s+2445\s+de\s+2025"),
}


def detect_norms(text: str) -> List[str]:
    """Return list of norm keys found in text."""
    return [k for k, pattern in _NORM_PATTERNS.items() if pattern.search(text)]


def load_pdf(path: Path) -> List[Document]:
    """
    Load a single PDF file.

    Returns one Document per page with metadata:
      - source: str (filename)
      - page: int (1-indexed)
      - total_pages: int
      - norms_cited: List[str]
    """
    docs: List[Document] = []
    try:
        with fitz.open(str(path)) as pdf:
            total = len(pdf)
            for i, page in enumerate(pdf):
                text = page.get_text("text").strip()
                if not text:
                    continue
                metadata: Dict[str, Any] = {
                    "source": path.name,
                    "page": i + 1,
                    "total_pages": total,
                    "norms_cited": detect_norms(text),
                }
                docs.append(Document(page_content=text, metadata=metadata))
    except Exception as exc:
        print(f"[WARN] Failed to load {path.name}: {exc}")
    return docs


def load_corpus(docs_dir: Path = DOCS_DIR) -> List[Document]:
    """
    Load every PDF in docs_dir.
    Raises FileNotFoundError if dir does not exist.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")
    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    all_docs: List[Document] = []
    for path in pdf_paths:
        all_docs.extend(load_pdf(path))
    print(f"[INFO] Loaded {len(all_docs)} pages from {len(pdf_paths)} PDFs.")
    return all_docs
```

**Dependencies:** Step 1.1.

---

### Step 2.2 — SemanticChunker implementation

**Files to create:** `src/ingestion/semantic_chunker.py`

**Effort:** 3 hours | **Risk:** Medium (threshold tuning needed for Spanish legal text)

```python
"""
src/ingestion/semantic_chunker.py

Wraps LangChain's SemanticChunker with project-specific defaults.
Uses a multilingual sentence-transformer optimised for Spanish.
"""
from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import (
    EMBEDDING_MODEL,
    CHUNK_BREAKPOINT_TYPE,
    CHUNK_BREAKPOINT_THRESHOLD,
)


def build_semantic_chunker() -> SemanticChunker:
    """
    Returns a configured SemanticChunker instance.

    breakpoint_threshold_type options:
        "percentile"    — split when distance > Nth percentile of all distances
        "standard_deviation" — split at mean + N*std
        "interquartile" — IQR-based
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=CHUNK_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=CHUNK_BREAKPOINT_THRESHOLD,
    )
    return chunker


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of page-level Documents into semantic chunks.
    Preserves and propagates source metadata onto each chunk.
    """
    chunker = build_semantic_chunker()
    chunks: List[Document] = []
    for doc in documents:
        split_docs = chunker.create_documents(
            [doc.page_content],
            metadatas=[doc.metadata],
        )
        chunks.extend(split_docs)
    print(f"[INFO] Produced {len(chunks)} semantic chunks from {len(documents)} pages.")
    return chunks
```

**Dependencies:** Step 2.1.

---

### Step 2.3 — Vector store indexing

**Files to create:** `src/ingestion/vector_store.py`

**Effort:** 2 hours | **Risk:** Low

```python
"""
src/ingestion/vector_store.py

Builds or loads a ChromaDB vector store from semantic chunks.
Exposes a retriever with MMR search.
"""
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    MMR_LAMBDA,
)


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Indexes chunks into ChromaDB and persists to disk.
    Safe to call multiple times — uses existing collection if present.
    """
    embeddings = _get_embeddings()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )
    print(f"[INFO] Vector store built with {store._collection.count()} documents.")
    return store


def load_vector_store() -> Chroma:
    """Load an already-built Chroma collection from disk."""
    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            "Vector store not found. Run build_vector_store() first."
        )
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


def get_mmr_retriever(store: Chroma) -> VectorStoreRetriever:
    """
    Returns a retriever using Maximum Marginal Relevance search.
    Fetches 2*k candidates then selects k diverse results.
    """
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": RETRIEVAL_K * 2,
            "lambda_mult": MMR_LAMBDA,
        },
    )


# --- Entry point for first-time indexing -----------------------------------
def run_ingestion_pipeline() -> Chroma:
    """Full pipeline: load PDFs -> chunk -> index. Called once."""
    from src.ingestion.pdf_loader import load_corpus
    from src.ingestion.semantic_chunker import chunk_documents
    docs = load_corpus()
    chunks = chunk_documents(docs)
    return build_vector_store(chunks)
```

**Dependencies:** Steps 2.1, 2.2.

---

## Phase 3: Knowledge Graph Integration (Days 5–7)

### Step 3.1 — RDFLib ontology manager

**Files to create:** `src/kg/ontology_manager.py`

**Effort:** 2 hours | **Risk:** Low

```python
"""
src/kg/ontology_manager.py

Loads the OWL ontology with RDFLib and exposes
SPARQL SELECT queries as typed Python functions.
"""
from pathlib import Path
from typing import List, Dict, Any

from rdflib import Graph, Namespace, URIRef
from rdflib.plugins.sparql import prepareQuery

from src.config import ONTOLOGY_PATH, ONTOLOGY_BASE_URI


INS = Namespace(ONTOLOGY_BASE_URI)

_INITNS = {
    "ins": INS,
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def load_graph(path: Path = ONTOLOGY_PATH) -> Graph:
    """Parse the Turtle ontology file into an RDFLib Graph."""
    g = Graph()
    g.parse(str(path), format="turtle")
    print(f"[INFO] Ontology loaded: {len(g)} triples.")
    return g


def query_graph(graph: Graph, sparql: str) -> List[Dict[str, Any]]:
    """
    Execute a SPARQL SELECT query on the local RDFLib graph.
    Returns a list of dicts {variable_name: value_str}.
    """
    result = graph.query(sparql, initNs=_INITNS)
    rows = []
    for row in result:
        rows.append({str(var): str(row[var]) for var in result.vars})
    return rows
```

---

### Step 3.2 — GraphDB SPARQL client

**Files to create:** `src/kg/graphdb_client.py`

**Effort:** 3 hours | **Risk:** Medium (network dependency on local GraphDB)

```python
"""
src/kg/graphdb_client.py

Thin wrapper around SPARQLWrapper for read/write against GraphDB.
Provides helpers for SELECT, UPDATE, and SPARQL-Fed queries.
"""
from typing import List, Dict, Any

from SPARQLWrapper import SPARQLWrapper, JSON, POST, DIGEST

from src.config import GRAPHDB_SPARQL_ENDPOINT, GRAPHDB_UPDATE_ENDPOINT

_INS = "http://www.unal.edu.co/ontologies/insolvencia#"


class GraphDBClient:
    """Stateless SPARQL client for the insolvencia repository."""

    def __init__(
        self,
        query_endpoint: str = GRAPHDB_SPARQL_ENDPOINT,
        update_endpoint: str = GRAPHDB_UPDATE_ENDPOINT,
    ):
        self._query_ep = query_endpoint
        self._update_ep = update_endpoint

    # ------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------
    def select(self, sparql: str) -> List[Dict[str, Any]]:
        """Execute a SPARQL SELECT and return list of binding dicts."""
        wrapper = SPARQLWrapper(self._query_ep)
        wrapper.setQuery(sparql)
        wrapper.setReturnFormat(JSON)
        results = wrapper.queryAndConvert()
        bindings = results["results"]["bindings"]
        return [
            {k: v["value"] for k, v in row.items()}
            for row in bindings
        ]

    # ------------------------------------------------------------------
    # UPDATE (INSERT / DELETE)
    # ------------------------------------------------------------------
    def update(self, sparql_update: str) -> None:
        """Execute a SPARQL UPDATE (INSERT DATA / DELETE DATA)."""
        wrapper = SPARQLWrapper(self._update_ep)
        wrapper.setMethod(POST)
        wrapper.setQuery(sparql_update)
        wrapper.query()

    # ------------------------------------------------------------------
    # Convenience query methods used by the KG tool
    # ------------------------------------------------------------------
    def get_procedures_for_debtor(self, debtor_name: str) -> List[Dict]:
        query = f"""
        PREFIX ins: <{_INS}>
        SELECT ?proc ?type ?fecha WHERE {{
            ?deudor ins:nombreRazonSocial "{debtor_name}"^^xsd:string .
            ?deudor ins:iniciaEn ?proc .
            ?proc a ?type .
            OPTIONAL {{ ?proc ins:fechaAdmision ?fecha }}
        }}
        ORDER BY ?fecha
        LIMIT 10
        """
        return self.select(query)

    def get_norms_regulating_procedure(self, proc_uri: str) -> List[Dict]:
        query = f"""
        PREFIX ins: <{_INS}>
        SELECT ?norma ?numero ?anio WHERE {{
            <{proc_uri}> ins:estaReguladoPor ?norma .
            ?norma ins:numeroNorma ?numero .
            OPTIONAL {{ ?norma ins:anoExpedicion ?anio }}
        }}
        ORDER BY DESC(?anio)
        """
        return self.select(query)

    def get_creditors_by_type(self, creditor_type: str = "AcreedorPrivilegiado") -> List[Dict]:
        query = f"""
        PREFIX ins: <{_INS}>
        SELECT ?acreedor ?nombre WHERE {{
            ?acreedor a ins:{creditor_type} .
            OPTIONAL {{ ?acreedor ins:nombreRazonSocial ?nombre }}
        }}
        LIMIT 20
        """
        return self.select(query)
```

---

### Step 3.3 — SPARQL query library

**Files to create:** `ontology/sparql/select_queries.sparql`, `filter_queries.sparql`, `update_queries.sparql`, `inference_cases.sparql`

**Effort:** 4 hours | **Risk:** Low

#### `select_queries.sparql` — SELECT with ORDER BY and LIMIT

```sparql
# Query 1: All insolvency procedures supervised by Superintendencia de Sociedades
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?proc ?tipo ?fecha
WHERE {
    ?proc a ?tipo .
    ?tipo rdfs:subClassOf ins:ProcedimientoInsolvencia .
    ?proc ins:esSupervisadoPor ins:SuperintendenciaSociedades .
    OPTIONAL { ?proc ins:fechaAdmision ?fecha }
}
ORDER BY DESC(?fecha)
LIMIT 10

---

# Query 2: Debtors with their creditors and obligation types
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?deudor ?nombreDeudor ?acreedor ?nombreAcreedor ?tipoAcreedor
WHERE {
    ?deudor a ?td .
    ?td rdfs:subClassOf ins:Deudor .
    ?deudor ins:tieneAcreedor ?acreedor .
    ?acreedor a ?tipoAcreedor .
    OPTIONAL { ?deudor ins:nombreRazonSocial ?nombreDeudor }
    OPTIONAL { ?acreedor ins:nombreRazonSocial ?nombreAcreedor }
}
ORDER BY ?nombreDeudor

---

# Query 3: All legal norms with year, ordered chronologically
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?norma ?tipo ?numero ?anio
WHERE {
    ?norma a ?tipo .
    ?tipo rdfs:subClassOf ins:NormaLegal .
    ?norma ins:numeroNorma ?numero .
    OPTIONAL { ?norma ins:anoExpedicion ?anio }
}
ORDER BY ?anio
LIMIT 20
```

#### `filter_queries.sparql` — SELECT with FILTER

```sparql
# Query 4: FILTER — procedures admitted after a given date
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?proc ?fecha
WHERE {
    ?proc a ?tipo .
    ?tipo rdfs:subClassOf ins:ProcedimientoInsolvencia .
    ?proc ins:fechaAdmision ?fecha .
    FILTER (?fecha > "2022-01-01"^^xsd:date)
}
ORDER BY ?fecha

---

# Query 5: FILTER — norms whose number contains "1116"
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?norma ?numero
WHERE {
    ?norma a ?tipo .
    ?tipo rdfs:subClassOf ins:NormaLegal .
    ?norma ins:numeroNorma ?numero .
    FILTER (CONTAINS(?numero, "1116"))
}

---

# Query 6: FILTER — creditors whose name contains "Banco" (case-insensitive)
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?acreedor ?nombre
WHERE {
    ?acreedor a ?tipo .
    ?tipo rdfs:subClassOf ins:Acreedor .
    ?acreedor ins:nombreRazonSocial ?nombre .
    FILTER (CONTAINS(LCASE(?nombre), "banco"))
}
ORDER BY ?nombre
LIMIT 10
```

#### `update_queries.sparql` — INSERT DATA / DELETE DATA

```sparql
# Update 1: Add a new reorganization procedure
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
INSERT DATA {
    ins:ReorgEjemplo3 a ins:Reorganizacion ;
        ins:fechaAdmision "2025-06-01"^^xsd:date ;
        ins:estaReguladoPor ins:Ley1116_2006 ;
        ins:esSupervisadoPor ins:SuperintendenciaSociedades .
}

---

# Update 2: Link debtor to new procedure
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
INSERT DATA {
    ins:MariaLopez ins:iniciaEn ins:ReorgEjemplo3 .
}

---

# Update 3: Delete a provisional individual
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
DELETE DATA {
    ins:JuntaEjemplo2 a ins:JuntaAcreedores .
}
```

#### `inference_cases.sparql` — 5 inference scenarios

```sparql
# Inference Case 1: SubClass reasoning
# PersonaJuridica subClassOf Deudor — query for all Deudor instances
# (should return both PersonaJuridica and PersonaNatural instances with inference ON)
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?deudor WHERE {
    ?deudor a ins:Deudor .
}

---

# Inference Case 2: inverseOf — from Acreedor navigate to Deudor
# esAcreedorDe owl:inverseOf tieneAcreedor
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?acreedor ?deudor WHERE {
    ?acreedor ins:esAcreedorDe ?deudor .
}

---

# Inference Case 3: esIniciadoPor owl:inverseOf iniciaEn
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?proc ?deudor WHERE {
    ?proc ins:esIniciadoPor ?deudor .
}

---

# Inference Case 4: SubProperty — tieneAcreedorPrivilegiado subPropertyOf tieneAcreedor
# Triples asserted via tieneAcreedorPrivilegiado should also be reachable via the
# superproperty tieneAcreedor when OWL inference is ON
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?deudor ?related WHERE {
    ?deudor ins:tieneAcreedor ?related .
}

---

# Inference Case 5: Union class — Deudor equivalentClass (PersonaJuridica OR PersonaNatural)
# Any PersonaJuridica or PersonaNatural should be inferred as Deudor
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?x WHERE {
    ?x a ins:Deudor .
}
```

---

### Step 3.4 — KG query tool for the agent

**Files to create:** `src/kg/sparql_tools.py`

**Effort:** 2 hours | **Risk:** Low

```python
"""
src/kg/sparql_tools.py

LangChain Tool wrappers that expose KG query capabilities
to the ReAct agent. Each tool is a callable that takes
a natural-language-like string argument and returns a string.
"""
from typing import Optional
from langchain_core.tools import tool

from src.kg.graphdb_client import GraphDBClient
from src.kg.ontology_manager import load_graph, query_graph

_client = GraphDBClient()
_graph = None   # lazy-loaded local RDFLib graph


def _get_graph():
    global _graph
    if _graph is None:
        _graph = load_graph()
    return _graph


@tool
def query_kg_procedures(debtor_name: str) -> str:
    """
    Query the Knowledge Graph for insolvency procedures related to a debtor.
    Input: the legal name (razon social) of the debtor company.
    Returns: formatted string with procedure types and admission dates.
    """
    results = _client.get_procedures_for_debtor(debtor_name)
    if not results:
        return f"No procedures found for debtor: {debtor_name}"
    lines = [f"Procedures for '{debtor_name}':"]
    for r in results:
        lines.append(f"  - {r.get('type','')} admitted on {r.get('fecha','unknown')}")
    return "\n".join(lines)


@tool
def query_kg_norms(keyword: str) -> str:
    """
    Search the Knowledge Graph for legal norms (Ley/Decreto) matching a keyword.
    Input: keyword such as '1116', '2006', 'decreto', or 'reorganizacion'.
    Returns: formatted list of matching norms.
    """
    # Sanitize input to prevent SPARQL injection
    safe_keyword = keyword.replace('"', '\\"').replace("'", "\\'").replace("\\", "\\\\")
    sparql = f"""
    PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?norma ?numero ?anio WHERE {{
        ?norma a ?tipo .
        ?tipo rdfs:subClassOf ins:NormaLegal .
        ?norma ins:numeroNorma ?numero .
        OPTIONAL {{ ?norma ins:anoExpedicion ?anio }}
        FILTER (CONTAINS(LCASE(STR(?norma)), LCASE("{safe_keyword}"))
             || CONTAINS(LCASE(?numero), LCASE("{safe_keyword}")))
    }}
    ORDER BY DESC(?anio)
    LIMIT 10
    """
    results = _client.select(sparql)
    if not results:
        return f"No norms found matching: {keyword}"
    lines = [f"Norms matching '{keyword}':"]
    for r in results:
        lines.append(f"  - Norm {r.get('numero','')} ({r.get('anio','')}): {r.get('norma','')}")
    return "\n".join(lines)


@tool
def query_kg_creditor_type(creditor_type: str) -> str:
    """
    Retrieve creditors from the Knowledge Graph by type.
    Input: 'AcreedorPrivilegiado' or 'AcreedorOrdinario'.
    Returns: formatted list of creditor names.
    """
    results = _client.get_creditors_by_type(creditor_type)
    if not results:
        return f"No creditors of type {creditor_type} found."
    lines = [f"Creditors of type '{creditor_type}':"]
    for r in results:
        lines.append(f"  - {r.get('nombre', r.get('acreedor',''))}")
    return "\n".join(lines)


@tool
def sparql_select_raw(sparql_query: str) -> str:
    """
    Execute an arbitrary SPARQL SELECT query against the local ontology graph.
    Use this only when other KG tools are insufficient.
    Input: valid SPARQL SELECT query string.
    Returns: JSON-like string of results.
    """
    try:
        rows = query_graph(_get_graph(), sparql_query)
        if not rows:
            return "Query returned no results."
        return "\n".join(str(r) for r in rows[:20])
    except Exception as e:
        return f"SPARQL error: {e}"


KG_TOOLS = [
    query_kg_procedures,
    query_kg_norms,
    query_kg_creditor_type,
    sparql_select_raw,
]
```

**Dependencies:** Steps 3.1, 3.2.

---

## Phase 4: Query Transformation (Days 7–8)

### Step 4.1 — HyDE implementation

**Files to create:** `src/query/hyde.py`

**Effort:** 2 hours | **Risk:** Low

```python
"""
src/query/hyde.py

Hypothetical Document Embeddings (HyDE).
Given a user query, generate a hypothetical answer document,
then embed THAT document to retrieve more semantically aligned chunks.
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un experto en derecho de insolvencia empresarial colombiano. "
     "Cuando recibas una pregunta, escribe un fragmento de texto legal "
     "(100-150 palabras) que RESPONDERÍA esa pregunta, como si fuera "
     "extraído de un documento jurídico real. No respondas directamente; "
     "genera el texto hipotético."),
    ("human", "{question}"),
])


def build_hyde_chain(llm: BaseChatModel) -> RunnableSequence:
    """
    Returns a chain: question -> hypothetical_document_string.
    Usage: hypothetical_text = chain.invoke({"question": user_query})
    """
    return _HYDE_PROMPT | llm | StrOutputParser()


def apply_hyde(llm: BaseChatModel, question: str) -> str:
    """Convenience wrapper. Returns the hypothetical document text."""
    chain = build_hyde_chain(llm)
    return chain.invoke({"question": question})
```

---

### Step 4.2 — Query decomposition

**Files to create:** `src/query/decomposer.py`

**Effort:** 2 hours | **Risk:** Low

```python
"""
src/query/decomposer.py

Decomposes a complex multi-hop legal question into 2-4 simpler sub-questions.
Each sub-question is retrieved independently; answers are merged.
"""
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser


_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un experto en análisis de preguntas legales. "
     "Descompón la siguiente pregunta compleja sobre derecho de insolvencia "
     "colombiana en 2-4 sub-preguntas simples y específicas. "
     "Devuelve ÚNICAMENTE las sub-preguntas separadas por punto y coma (;). "
     "No incluyas numeración ni texto adicional."),
    ("human", "{question}"),
])


def build_decomposer(llm: BaseChatModel):
    """
    Returns chain: question -> List[str] of sub-questions.
    """
    parser = CommaSeparatedListOutputParser()

    # Override separator to semicolon since legal Spanish uses commas in text
    class SemicolonListParser(CommaSeparatedListOutputParser):
        def parse(self, text: str) -> List[str]:
            return [q.strip() for q in text.split(";") if q.strip()]

    return _DECOMPOSE_PROMPT | llm | SemicolonListParser()


def decompose_query(llm: BaseChatModel, question: str) -> List[str]:
    """
    Returns list of sub-questions derived from the original question.
    Falls back to [question] if decomposition fails or returns only 1 item.
    """
    try:
        chain = build_decomposer(llm)
        parts = chain.invoke({"question": question})
        return parts if len(parts) > 1 else [question]
    except Exception:
        return [question]
```

---

### Step 4.3 — Query router

**Files to create:** `src/query/router.py`

**Effort:** 2 hours | **Risk:** Low

```python
"""
src/query/router.py

Classifies the user query to decide which retrieval path to use:
  "vector"  — pure semantic search (factual, definition questions)
  "kg"      — Knowledge Graph + SPARQL (entity, norm, procedure lookup)
  "hybrid"  — both vector + KG (complex multi-hop)
"""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


QueryRoute = Literal["vector", "kg", "hybrid"]

_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Clasifica la siguiente pregunta sobre insolvencia empresarial colombiana. "
     "Responde ÚNICAMENTE con una de estas palabras: vector, kg, hybrid.\n"
     "- vector: preguntas sobre conceptos, definiciones, artículos de ley.\n"
     "- kg: preguntas sobre entidades específicas, acreedores, empresas, normas.\n"
     "- hybrid: preguntas complejas que requieren ambos tipos de información."),
    ("human", "{question}"),
])


def build_router(llm: BaseChatModel):
    return _ROUTER_PROMPT | llm | StrOutputParser()


def route_query(llm: BaseChatModel, question: str) -> QueryRoute:
    """Returns the retrieval route for the given question."""
    chain = build_router(llm)
    raw = chain.invoke({"question": question}).strip().lower()
    if raw in ("vector", "kg", "hybrid"):
        return raw  # type: ignore
    return "hybrid"  # safe default
```

**Dependencies:** Step 1.1 (config, LLM settings).

---

## Phase 5: Agent Architecture (Days 8–10)

### Step 5.1 — LangGraph state definition

**Files to create:** `src/agent/state.py`

**Effort:** 1 hour | **Risk:** Low

```python
"""
src/agent/state.py

TypedDict definition for the LangGraph agent state.
Every node reads from and writes to this shared state.
"""
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Conversation thread (immutable append-only via add_messages)
    messages: Annotated[List[BaseMessage], add_messages]

    # The original user question
    question: str

    # Transformed / decomposed sub-questions
    transformed_queries: List[str]

    # Retrieval path decided by router
    route: str  # "vector" | "kg" | "hybrid"

    # Retrieved context chunks (page_content strings)
    context_chunks: List[str]

    # KG query results (formatted strings)
    kg_results: List[str]

    # The current generated answer
    answer: str

    # Reflection / critique from the critic node
    critique: Optional[str]

    # Number of retry iterations completed
    retry_count: int

    # Whether the reflection node approved the answer
    answer_approved: bool

    # Whether the web fallback was triggered
    used_web_fallback: bool

    # True after web fallback has been used (prevents infinite retry loops)
    web_fallback_used: bool

    # Final sources cited in the response
    sources: List[str]
```

---

### Step 5.2 — Tools for the agent

**Files to create:** `src/agent/tools.py`

**Effort:** 2 hours | **Risk:** Low

```python
"""
src/agent/tools.py

Assembles the complete tool list available to the ReAct agent:
  1. vector_search  — MMR retrieval from ChromaDB
  2. kg_procedures  — KG query for insolvency procedures
  3. kg_norms       — KG query for legal norms
  4. kg_creditors   — KG query for creditor types
  5. sparql_raw     — raw SPARQL SELECT on local ontology
  6. web_search     — DuckDuckGo fallback (triggered after 3 retries)
"""
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore

from src.kg.sparql_tools import KG_TOOLS


def build_vector_search_tool(vector_store: VectorStore):
    @tool
    def vector_search(query: str) -> str:
        """
        Search the insolvency law document corpus using semantic similarity.
        Input: a specific question or phrase about Colombian insolvency law.
        Returns: up to 6 relevant text passages.
        """
        docs = vector_store.similarity_search(query, k=6)
        if not docs:
            return "No relevant documents found."
        passages = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            pg = doc.metadata.get("page", "?")
            passages.append(f"[{i}] ({src}, p.{pg})\n{doc.page_content[:600]}")
        return "\n\n".join(passages)
    return vector_search


def build_all_tools(vector_store: VectorStore):
    """Return the complete tool list for the agent."""
    from src.agent.web_fallback import web_search_tool
    return [build_vector_search_tool(vector_store)] + KG_TOOLS + [web_search_tool]
```

---

### Step 5.3 — Agent nodes

**Files to create:** `src/agent/nodes.py`

**Effort:** 5 hours | **Risk:** High (most complex component)

```python
"""
src/agent/nodes.py

LangGraph node functions implementing the ReAct + Reflecting pattern.
Each function signature: (state: AgentState) -> dict[str, Any]
The returned dict is merged into the shared state.
"""
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.config import LLM_MODEL, LLM_TEMPERATURE, MAX_RETRIES
from src.query.hyde import apply_hyde
from src.query.decomposer import decompose_query
from src.query.router import route_query


def _make_llm(temperature: float = LLM_TEMPERATURE) -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


# ---------------------------------------------------------------------------
# Node 1: query_transform
# Applies HyDE + decomposition and determines route
# ---------------------------------------------------------------------------
def query_transform_node(state: AgentState) -> Dict[str, Any]:
    llm = _make_llm()
    question = state["question"]

    # Route first (cheap classification)
    route = route_query(llm, question)

    # Sub-question decomposition
    sub_questions = decompose_query(llm, question)

    # HyDE for each sub-question (used in hybrid/vector paths)
    if route in ("vector", "hybrid"):
        hyde_docs = [apply_hyde(llm, q) for q in sub_questions]
        transformed = hyde_docs
    else:
        transformed = sub_questions

    return {
        "route": route,
        "transformed_queries": transformed,
        "messages": [AIMessage(content=f"[Router] Path: {route}. Sub-questions: {sub_questions}")],
    }


# ---------------------------------------------------------------------------
# Node 2: retrieve
# Calls vector store and/or KG tools based on route
# ---------------------------------------------------------------------------
def retrieve_node(state: AgentState, tools_map: dict) -> Dict[str, Any]:
    route = state["route"]
    queries = state["transformed_queries"]
    chunks: List[str] = []
    kg_results: List[str] = []

    vector_tool = tools_map.get("vector_search")
    kg_proc_tool = tools_map.get("query_kg_procedures")
    kg_norm_tool = tools_map.get("query_kg_norms")

    for q in queries:
        if route in ("vector", "hybrid") and vector_tool:
            result = vector_tool.invoke(q)
            chunks.append(result)

        if route in ("kg", "hybrid"):
            # Extract potential entity / norm from query (heuristic)
            if kg_norm_tool and any(w in q.lower() for w in ["ley", "decreto", "norma", "artículo"]):
                kg_results.append(kg_norm_tool.invoke(q))
            if kg_proc_tool and any(w in q.lower() for w in ["empresa", "deudor", "reorganiz", "liquidac"]):
                kg_results.append(kg_proc_tool.invoke(q))

    return {
        "context_chunks": chunks,
        "kg_results": kg_results,
    }


# ---------------------------------------------------------------------------
# Node 3: generate
# Synthesises answer from context
# ---------------------------------------------------------------------------
_GENERATE_SYSTEM = """Eres un experto en derecho de insolvencia empresarial colombiana.
Responde la pregunta del usuario usando ÚNICAMENTE la información proporcionada
en el contexto. Si el contexto no es suficiente, indícalo explícitamente.
Cita siempre las fuentes (nombre del documento y página)."""


def generate_node(state: AgentState) -> Dict[str, Any]:
    llm = _make_llm()
    context = "\n\n---\n\n".join(state.get("context_chunks", []))
    kg_context = "\n\n".join(state.get("kg_results", []))

    full_context = f"DOCUMENTOS:\n{context}\n\nGRAFO DE CONOCIMIENTO:\n{kg_context}"

    messages = [
        SystemMessage(content=_GENERATE_SYSTEM),
        HumanMessage(content=f"CONTEXTO:\n{full_context}\n\nPREGUNTA: {state['question']}"),
    ]
    response = llm.invoke(messages)
    return {
        "answer": response.content,
        "messages": [response],
    }


# ---------------------------------------------------------------------------
# Node 4: reflect (critic)
# Evaluates the answer and decides if it is acceptable
# ---------------------------------------------------------------------------
_REFLECT_SYSTEM = """Eres un crítico experto en derecho colombiano.
Evalúa la respuesta proporcionada en términos de:
1. Fidelidad al contexto (no alucina hechos)
2. Completitud (responde la pregunta completamente)
3. Precisión legal (cita correctamente las normas)

Si la respuesta ES satisfactoria, responde: APROBADO
Si NO es satisfactoria, responde: RECHAZADO: <explicación breve de qué falta>"""


def reflect_node(state: AgentState) -> Dict[str, Any]:
    llm = _make_llm(temperature=0.0)
    messages = [
        SystemMessage(content=_REFLECT_SYSTEM),
        HumanMessage(content=(
            f"PREGUNTA: {state['question']}\n\n"
            f"RESPUESTA GENERADA:\n{state['answer']}\n\n"
            f"CONTEXTO USADO:\n{' '.join(state.get('context_chunks', []))[:2000]}"
        )),
    ]
    critique = llm.invoke(messages).content.strip()
    approved = critique.startswith("APROBADO")
    return {
        "critique": critique,
        "answer_approved": approved,
        "retry_count": state["retry_count"] + (0 if approved else 1),
        "messages": [AIMessage(content=f"[Critic] {critique}")],
    }


# ---------------------------------------------------------------------------
# Node 5: web_fallback
# Called after MAX_RETRIES failed reflection cycles
# ---------------------------------------------------------------------------
def web_fallback_node(state: AgentState) -> Dict[str, Any]:
    """
    Runs a DuckDuckGo web search and appends results to context.
    Sets web_fallback_used=True to prevent reflection from triggering
    another retry cycle after the web fallback path.
    """
    from src.agent.web_fallback import search_web
    results = search_web(state["question"])
    return {
        "context_chunks": state.get("context_chunks", []) + [results],
        "used_web_fallback": True,
        "web_fallback_used": True,
        "messages": [AIMessage(content="[WebFallback] Retrieved supplementary web context.")],
    }
```

---

### Step 5.4 — LangGraph graph definition

**Files to create:** `src/agent/graph.py`

**Effort:** 3 hours | **Risk:** Medium

```python
"""
src/agent/graph.py

Assembles the LangGraph StateGraph implementing the full
ReAct + Reflecting agentic loop with web fallback.

Flow:
  START
    -> query_transform
    -> retrieve
    -> react_agent       (LangGraph prebuilt ReAct with tools)
    -> reflect
    -> [approved?]
        YES -> END
        NO  -> [retry_count >= MAX_RETRIES?]
            YES -> web_fallback -> generate -> END
            NO  -> retrieve (retry loop)
"""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.nodes import (
    query_transform_node,
    retrieve_node,
    generate_node,
    reflect_node,
    web_fallback_node,
)
from src.agent.tools import build_all_tools
from src.config import LLM_MODEL, MAX_RETRIES
from src.ingestion.vector_store import load_vector_store


def should_retry(state: AgentState) -> str:
    """Conditional edge: retry, fallback, or end.

    After web_fallback has been used, always route to END to prevent
    an infinite retry loop (web_fallback -> generate -> reflect -> web_fallback ...).
    """
    if state.get("web_fallback_used", False):
        return "end"  # web fallback already ran — always terminate
    if state["answer_approved"]:
        return "end"
    if state["retry_count"] >= MAX_RETRIES:
        return "web_fallback"
    return "retrieve"


def build_graph():
    """Build and compile the LangGraph agent graph."""
    vector_store = load_vector_store()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    all_tools = build_all_tools(vector_store)
    tools_map = {t.name: t for t in all_tools}

    # Create the inner ReAct agent (handles tool calling loop)
    react_agent_runnable = create_react_agent(llm, all_tools)

    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("query_transform", query_transform_node)
    builder.add_node("retrieve", lambda s: retrieve_node(s, tools_map))
    builder.add_node("react_agent", react_agent_runnable)
    builder.add_node("generate", generate_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("web_fallback", web_fallback_node)

    # Edges
    builder.add_edge(START, "query_transform")
    builder.add_edge("query_transform", "retrieve")
    builder.add_edge("retrieve", "react_agent")
    builder.add_edge("react_agent", "generate")
    builder.add_edge("generate", "reflect")

    builder.add_conditional_edges(
        "reflect",
        should_retry,
        {
            "end": END,
            "web_fallback": "web_fallback",
            "retrieve": "retrieve",
        },
    )
    builder.add_edge("web_fallback", "generate")

    graph = builder.compile()
    return graph


def run_query(question: str) -> dict:
    """Main entry point: invoke the agent graph with a question."""
    graph = build_graph()
    initial_state: AgentState = {
        "messages": [],
        "question": question,
        "transformed_queries": [],
        "route": "hybrid",
        "context_chunks": [],
        "kg_results": [],
        "answer": "",
        "critique": None,
        "retry_count": 0,
        "answer_approved": False,
        "used_web_fallback": False,
        "web_fallback_used": False,
        "sources": [],
    }
    final_state = graph.invoke(initial_state)
    return {
        "answer": final_state["answer"],
        "sources": final_state.get("sources", []),
        "used_web_fallback": final_state.get("used_web_fallback", False),
        "retry_count": final_state.get("retry_count", 0),
    }
```

---

### Step 5.5 — Web fallback

**Files to create:** `src/agent/web_fallback.py`

**Effort:** 1 hour | **Risk:** Low

```python
"""
src/agent/web_fallback.py

DuckDuckGo-based web search fallback used when the KG+vector
retrieval fails to produce an approved answer after MAX_RETRIES.
"""
from langchain_core.tools import tool
from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5) -> str:
    """
    Search DuckDuckGo for Colombian insolvency law information.
    Returns formatted string of result snippets.
    """
    search_query = f"insolvencia empresarial Colombia Ley 1116 2006 {query}"
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(search_query, max_results=max_results):
            results.append(f"[{r['title']}] {r['body']}\nURL: {r['href']}")
    return "\n\n".join(results) if results else "No web results found."


@tool
def web_search_tool(query: str) -> str:
    """
    Fallback web search for Colombian insolvency law questions.
    Use ONLY when vector store and KG retrieval are insufficient.
    Input: the user's question or a refined search query.
    """
    return search_web(query)
```

**Dependencies:** Steps 5.1–5.4.

---

## Phase 6: LangSmith & Evaluation (Days 10–11)

### Step 6.1 — LangSmith tracing setup

**Files to create:** `src/tracing/langsmith_setup.py`

**Effort:** 1 hour | **Risk:** Low

```python
"""
src/tracing/langsmith_setup.py

Configures LangSmith tracing for all LangGraph nodes.
Must be imported (and configure_langsmith() called) before
any LangChain/LangGraph objects are created.
"""
import os
from src.config import (
    LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT,
    LANGCHAIN_TRACING_V2,
)


def configure_langsmith() -> None:
    """
    Set environment variables required by LangSmith SDK.
    Call this once at application startup (before any chain is built).
    """
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    print(f"[LangSmith] Tracing enabled. Project: {LANGCHAIN_PROJECT}")


def get_run_url(run_id: str) -> str:
    """Return the LangSmith UI URL for a given run ID."""
    return f"https://smith.langchain.com/o/default/projects/{LANGCHAIN_PROJECT}/runs/{run_id}"
```

**Note:** Each LangGraph node is automatically traced by LangSmith when
`LANGCHAIN_TRACING_V2=true` is set. No manual instrumentation is needed
beyond calling `configure_langsmith()` at startup.

---

### Step 6.2 — Evaluation metrics

**Files to create:** `src/evaluation/metrics.py`

**Effort:** 4 hours | **Risk:** Medium (requires ground truth dataset)

```python
"""
src/evaluation/metrics.py

Implements Recall@k, Precision@k, MRR, and nDCG
for the retrieval component, plus answer-level RAGAS metrics.
"""
import math
from typing import List, Set, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """
    Recall@k = |relevant ∩ retrieved[:k]| / |relevant|
    Document IDs should be chunk IDs or source+page strings.
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(relevant & top_k) / len(relevant)


def precision_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Precision@k = |relevant ∩ retrieved[:k]| / k"""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / k


def reciprocal_rank(relevant: Set[str], retrieved: List[str]) -> float:
    """MRR contribution for a single query: 1/rank of first relevant result."""
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(queries_results: List[Dict]) -> float:
    """
    queries_results: list of {"relevant": set, "retrieved": list}
    Returns MRR over all queries.
    """
    rr_scores = [reciprocal_rank(q["relevant"], q["retrieved"]) for q in queries_results]
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def dcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Discounted Cumulative Gain @k."""
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Normalised DCG@k."""
    actual_dcg = dcg_at_k(relevant, retrieved, k)
    ideal_retrieved = list(relevant) + [f"__non_relevant_{i}" for i in range(k)]
    ideal_dcg = dcg_at_k(relevant, ideal_retrieved, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Answer-level metrics (wrapper around RAGAS)
# ---------------------------------------------------------------------------

def compute_ragas_metrics(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dict[str, float]:
    """
    Compute RAGAS answer_relevancy and faithfulness.
    Returns dict with metric names as keys.
    """
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
```

---

### Step 6.3 — LLM-as-judge

**Files to create:** `src/evaluation/llm_judge.py`

**Effort:** 2 hours | **Risk:** Low

```python
"""
src/evaluation/llm_judge.py

Uses an LLM to score generated answers on a 1-5 scale along
three dimensions: Relevance, Faithfulness, Legal Accuracy.
"""
from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un juez experto en derecho colombiano de insolvencia. "
     "Evalúa la respuesta en una escala de 1 a 5 en TRES dimensiones.\n"
     "Responde EXACTAMENTE en este formato (sin texto adicional):\n"
     "Relevancia: <1-5>\n"
     "Fidelidad: <1-5>\n"
     "Precision_Legal: <1-5>\n"
     "Justificacion: <una frase>"),
    ("human",
     "PREGUNTA: {question}\n\n"
     "RESPUESTA: {answer}\n\n"
     "CONTEXTO FUENTE (primeros 1500 chars):\n{context}"),
])


def llm_judge_score(
    llm: BaseChatModel,
    question: str,
    answer: str,
    context: str,
) -> Dict[str, float]:
    """
    Returns dict:
      {
        "relevancia": float,
        "fidelidad": float,
        "precision_legal": float,
        "justificacion": str,
        "promedio": float,
      }
    """
    chain = _JUDGE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "question": question,
        "answer": answer,
        "context": context[:1500],
    })

    scores: Dict[str, float] = {}
    for line in raw.strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            try:
                scores[key] = float(val)
            except ValueError:
                scores[key] = val  # type: ignore  (for justificacion)

    numeric = [v for k, v in scores.items() if isinstance(v, float)]
    scores["promedio"] = sum(numeric) / len(numeric) if numeric else 0.0
    return scores
```

**Dependencies:** Steps 5.1–5.5 (needs agent output to evaluate).

---

## Phase 7: Documentation & Delivery (Days 11–13)

### Step 7.1 — Technical report structure

**File to create:** `report/technical_report.md` (source for PDF export)

The report must cover these sections (export to PDF via Pandoc or Typst):

```
1. Introduccion
   1.1 Contexto y objetivos
   1.2 Corpus (50 PDFs — descripcion, fuentes, cobertura legal)

2. Arquitectura del Sistema
   2.1 Diagrama de flujo del agente (mermaid)
   2.2 Componentes principales

3. Ontologia OWL
   3.1 Clases y jerarquia
   3.2 Propiedades y restricciones
   3.3 Constructores logicos
   3.4 Individuos nombrados
   3.5 Casos de inferencia (5 casos documentados)

4. Ingestion y Vector Store
   4.1 Pipeline de carga de PDFs
   4.2 SemanticChunker — parametros y justificacion
   4.3 Embedding model — justificacion de eleccion

5. Knowledge Graph RAG
   5.1 Conexion GraphDB
   5.2 Consultas SPARQL documentadas (SELECT, FILTER, UPDATE)
   5.3 Herramienta KG para el agente

6. Transformacion de Consultas
   6.1 HyDE — implementacion y ejemplos
   6.2 Descomposicion de preguntas — ejemplos
   6.3 Router — criterios de clasificacion

7. Agente ReAct + Reflecting
   7.1 Grafo LangGraph (diagrama)
   7.2 Nodos y logica de cada nodo
   7.3 Ciclo de reflexion y reintento
   7.4 Fallback web

8. LangSmith
   8.1 Configuracion de trazabilidad
   8.2 Capturas de pantalla de trazas

9. Evaluacion
   9.1 Metricas de recuperacion (Recall@k, Precision@k, MRR, nDCG)
   9.2 Metricas de respuesta (RAGAS)
   9.3 LLM-as-judge — resultados
   9.4 Analisis de resultados

10. Conclusiones y trabajo futuro
```

---

### Step 7.2 — Code documentation checklist

Every module must have:
- Module-level docstring with purpose and usage example
- Function/class docstrings with Args and Returns
- Inline comments on non-obvious logic
- Type annotations on all public functions

Run `pydoc` or `pdoc` to generate HTML docs:
```bash
pdoc src/ --output-dir docs/api/
```

---

### Step 7.3 — Video preparation outline (10 min)

```
0:00 - 0:30  Introduction: problem statement + corpus overview
0:30 - 1:30  Ontology walkthrough in Protege (classes, properties, individuals)
1:30 - 3:00  GraphDB demo: load TTL, run 3 SPARQL queries live, show inference
3:00 - 4:30  Ingestion pipeline demo: run run_ingestion_pipeline(), show Chroma counts
4:30 - 5:30  Query transformation demo: HyDE + decomposition on a sample question
5:30 - 7:30  Agent demo: run_query() on 3 different questions, show LangSmith trace
7:30 - 8:30  Evaluation results: show metrics table, LLM-as-judge scores
8:30 - 9:30  Code walkthrough: LangGraph graph.py, nodes.py
9:30 - 10:00 Conclusions + Q&A prompt
```

---

## Dependency Map

```
Step 1.1 (config)
  └──> Step 1.2 (ontology.ttl)
        └──> Step 1.3 (GraphDB upload)
  └──> Step 2.1 (pdf_loader)
        └──> Step 2.2 (semantic_chunker)
              └──> Step 2.3 (vector_store)
  └──> Step 3.1 (ontology_manager)
        └──> Step 3.2 (graphdb_client)
              └──> Step 3.3 (SPARQL queries)
                    └──> Step 3.4 (sparql_tools)
  └──> Step 4.1 (hyde)
  └──> Step 4.2 (decomposer)
  └──> Step 4.3 (router)

Step 2.3 + Step 3.4 + Steps 4.1-4.3
  └──> Step 5.1 (state)
        └──> Step 5.2 (tools)
        └──> Step 5.3 (nodes)
        └──> Step 5.4 (graph)   ← integrates everything
        └──> Step 5.5 (web_fallback)

Step 5.4 (graph)
  └──> Step 6.1 (langsmith_setup)
  └──> Step 6.2 (metrics)
  └──> Step 6.3 (llm_judge)

All steps
  └──> Step 7.1 (technical report)
  └──> Step 7.2 (docs)
  └──> Step 7.3 (video)
```

---

## Risk Register

| Risk | Phase | Severity | Mitigation |
|---|---|---|---|
| GraphDB not running or unreachable | 1.3, 3.2 | High | Use RDFLib local graph as fallback; keep `sparql_select_raw` tool working offline |
| SemanticChunker produces too many/few chunks | 2.2 | Medium | Tune `breakpoint_threshold_amount` (try 85, 90, 95); inspect chunk count interactively in notebook |
| Spanish PDF OCR quality poor | 2.1 | Medium | Use `fitz` (PyMuPDF) as primary parser; fall back to `unstructured` for scanned docs |
| LangGraph version incompatibility | 5.4 | Medium | Pin `langgraph>=0.2,<0.3` in requirements; test with `langgraph.graph.StateGraph` API |
| RAGAS not supporting Spanish | 6.2 | Low | Use GPT-4o as the evaluation LLM within RAGAS (set `RAGAS_LLM=gpt-4o` env var) |
| LangSmith API rate limits | 6.1 | Low | Batch evaluation runs; use `langsmith.wrappers.wrap_openai` for sampling |
| OpenAI API costs | All phases | Medium | Use `gpt-4o-mini` for all non-evaluation calls; use `gpt-4o` only for LLM-as-judge |
| Ontology inference not working in GraphDB | 3.3 | Medium | Enable OWL2-RL ruleset in GraphDB repo config; verify with `owl:sameAs` test triple |
| Time pressure (13 days) | All | High | Follow phase order strictly; Phase 5 (agent) is the critical path — do not start Phase 6 until graph.py compiles |

---

## Quick-Start Commands

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill environment variables
cp .env.example .env
# Edit .env with your OpenAI and LangSmith keys

# 4. Run the ingestion pipeline (one-time, ~5-10 min)
python -c "from src.ingestion.vector_store import run_ingestion_pipeline; run_ingestion_pipeline()"

# 5. Upload ontology to GraphDB (GraphDB must be running)
curl -X POST -H "Content-Type: text/turtle" \
     --data-binary @ontology/insolvencia.ttl \
     http://localhost:7200/repositories/insolvencia/statements

# 6. Test a single agent query
python -c "
from src.tracing.langsmith_setup import configure_langsmith
configure_langsmith()
from src.agent.graph import run_query
result = run_query('¿Cuáles son los requisitos para iniciar un proceso de reorganizacion bajo la Ley 1116 de 2006?')
print(result['answer'])
"

# 7. Run evaluation metrics
python -m pytest tests/ -v
```

---

*Plan created: 2026-03-28 | Due: 2026-04-10 | Author: Implementation Planner*
