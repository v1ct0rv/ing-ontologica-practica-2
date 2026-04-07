# Knowledge Graph RAG — Insolvencia Empresarial Colombiana

**Universidad Nacional de Colombia — Sede Medellín**
**Ingeniería Ontológica (3010090) — Trabajo Práctico 2**
**Profesor:** Jaime Alberto Guzmán Luna

---

## Descripcion del Sistema

Sistema RAG (Retrieval-Augmented Generation) agéntico que responde preguntas complejas en lenguaje natural sobre **insolvencia empresarial colombiana** (Ley 1116 de 2006, Decreto 806 de 2020, y normativa relacionada).

El sistema integra:

- **50 documentos PDF** procesados con Chunking Semántico (4,525 chunks)
- **Ontología OWL** en formato Turtle con 16 clases, 18 propiedades y 41 individuos
- **Agente ReAct + Reflecting** orquestado con LangGraph (6 nodos)
- **Búsqueda MMR** (Maximal Marginal Relevance) en ChromaDB
- **Knowledge Graph RAG** combinando vector store + SPARQL sobre GraphDB
- **Transformación de consultas**: HyDE y Query Decomposition
- **Trazabilidad completa** con LangSmith
- **Evaluación**: Recall@k, Precision@k, MRR, nDCG, LLM como juez

---

## Estructura del Proyecto

```text
Practica 2/
├── main.py                     # Punto de entrada principal (CLI)
├── requirements.txt            # Dependencias Python
├── .env.example                # Template de variables de entorno
├── .env                        # Variables reales (no versionado)
│
├── content/
│   └── docs/                   # 50 PDFs del corpus
│
├── ontology/
│   ├── insolvencia.ttl         # Ontología OWL (Turtle)
│   └── sparql/
│       ├── select_queries.sparql
│       ├── filter_queries.sparql
│       ├── update_queries.sparql
│       └── inference_cases.sparql
│
├── src/
│   ├── config.py               # Configuración centralizada
│   ├── ingestion/
│   │   ├── pdf_loader.py       # Carga PDFs con PyMuPDF
│   │   ├── semantic_chunker.py # Chunking semántico (SemanticChunker)
│   │   └── vector_store.py     # ChromaDB + búsqueda MMR
│   ├── kg/
│   │   ├── ontology_manager.py # RDFLib — SPARQL sobre grafo local
│   │   ├── graphdb_client.py   # SPARQLWrapper → GraphDB remoto
│   │   └── sparql_tools.py     # 5 herramientas LangChain para el agente
│   ├── query/
│   │   ├── hyde.py             # HyDE (Hypothetical Document Embeddings)
│   │   ├── decomposer.py      # Descomposición de consultas multi-hop
│   │   └── router.py          # Clasificador de estrategia (DIRECT/HYDE/DECOMPOSE)
│   ├── agent/
│   │   ├── state.py            # AgentState (TypedDict, 13 campos)
│   │   ├── tools.py            # 7 herramientas del agente ReAct
│   │   ├── nodes.py            # 5 nodos LangGraph
│   │   ├── graph.py            # Grafo StateGraph compilado
│   │   └── web_fallback.py     # Búsqueda web DuckDuckGo (fallback)
│   ├── evaluation/
│   │   ├── metrics.py          # Recall@k, Precision@k, MRR, nDCG, RAGAS
│   │   └── llm_judge.py        # LLM como juez (3 dimensiones)
│   └── tracing/
│       └── langsmith_setup.py  # Configuración LangSmith
│
├── tests/
│   ├── test_metrics.py         # 15 tests de métricas de retrieval
│   └── test_query_transform.py # 11 tests (5 sin API, 6 con API)
│
├── notebooks/
│   └── demo.ipynb              # Demo interactivo completo
│
├── report/
│   └── technical_report.md     # Informe técnico (fuente para PDF)
│
└── docs/
    ├── REQUIREMENTS.md         # Requisitos detallados con checklists
    ├── ARCHITECTURE.md         # Arquitectura del sistema
    └── IMPLEMENTATION_PLAN.md  # Plan de implementación por fases
```

---

## Requisitos Previos

- **Python 3.11+** (probado con 3.14)
- **GraphDB Free** (opcional, para Knowledge Graph remoto — sin él funciona con RDFLib local)
- **API Keys**:
  - `OPENAI_API_KEY` — requerida para el LLM (GPT-4o-mini)
  - `LANGCHAIN_API_KEY` — requerida para trazabilidad LangSmith

---

## Instalación Paso a Paso

### 1. Crear entorno virtual e instalar dependencias

```bash
cd "Practica 2/"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env` con tus claves:

```dotenv
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=insolvencia-kg-rag
GRAPHDB_ENDPOINT=http://localhost:7200
GRAPHDB_REPOSITORY=insolvencia
LLM_MODEL=gpt-4o-mini
```

### 3. Ejecutar el pipeline de ingesta (primera vez)

Este paso carga los 50 PDFs, aplica chunking semántico y construye el vector store en ChromaDB. **Solo se ejecuta una vez** (~3-5 minutos).

```bash
python main.py --ingest
```

Resultado esperado:

```text
[INFO] Cargadas 1863 paginas de 50 PDFs.
[INFO] Generados 4525 chunks semanticos de 1863 paginas.
[INFO] Vector store construido con 4525 documentos.
[OK] Pipeline completado. 4525 chunks en ChromaDB.
```

> Si ya existe `.chroma_db/`, la ingesta se puede saltar. El vector store se carga automáticamente.

### 4. (Opcional) Configurar GraphDB

Si deseas usar el Knowledge Graph con GraphDB:

```bash
# Opción A: Docker
docker run -d --name graphdb -p 7200:7200 ontotext/graphdb:10.7.3

# Opción B: Instalación local desde https://www.ontotext.com/products/graphdb/
```

Luego crear el repositorio y subir la ontología (2 pasos):

```bash
# Paso 1: Crear el repositorio con ruleset OWL2-RL (necesario para inferencias)
curl -X POST http://localhost:7200/rest/repositories \
     -H "Content-Type: multipart/form-data" \
     -F "config=@ontology/graphdb_repo_config.ttl"

# Paso 2: Subir la ontología
curl -X POST -H "Content-Type: text/turtle" \
     --data-binary @ontology/insolvencia.ttl \
     http://localhost:7200/repositories/insolvencia/statements
```

**Alternativa visual:** Abrir `http://localhost:7200` en el navegador → Setup → Repositories → Create new repository → Repository ID: `insolvencia`, Ruleset: `OWL2-RL` → Create. Luego Import → RDF → Upload `ontology/insolvencia.ttl`.

**Verificar la carga:**

```bash
# Debe retornar un conteo > 300 triples
curl -s "http://localhost:7200/repositories/insolvencia?query=SELECT+(COUNT(*)+AS+%3Fc)+WHERE+%7B%3Fs+%3Fp+%3Fo%7D" \
     -H "Accept: application/sparql-results+json"
```

> **Sin GraphDB**, el sistema usa RDFLib como grafo local para consultas SPARQL. Las herramientas KG del agente funcionan igual. GraphDB es necesario únicamente para las **5 inferencias documentadas** (requieren razonador OWL2-RL).

---

## Uso del Sistema

### Consulta individual

```bash
python main.py "¿Qué es la cesación de pagos según la Ley 1116?"
```

Salida:

```text
ESTRATEGIA: DIRECT
REINTENTOS: 0
WEB FALLBACK: No

RESPUESTA:
------------------------------------------------------------
La cesación de pagos, según la Ley 1116 de 2006, se refiere...
------------------------------------------------------------

FUENTES (6):
  - ley insolvencia 2006.pdf, p.29
  - GUIA ORIENTACION final.pdf, p.56
  ...

CRITICA: APROBADO
```

### Modo interactivo

```bash
python main.py --interactive
```

```text
KNOWLEDGE GRAPH RAG — Insolvencia Empresarial Colombia
Escriba 'salir' para terminar
============================================================

> Pregunta: ¿Cuáles son los requisitos para la reorganización empresarial?

[Estrategia: DIRECT | Reintentos: 0 | Web: No]

Los requisitos para la reorganización empresarial según la Ley 1116...
```

### Evaluar métricas de retrieval

```bash
python main.py --evaluate
```

### Correr tests

```bash
python -m pytest tests/ -v
```

Resultado esperado: `20 passed, 6 skipped` (los 6 skipped requieren OPENAI_API_KEY).

### Notebook demo

```bash
jupyter notebook notebooks/demo.ipynb
```

El notebook demuestra paso a paso:

1. Vector store y búsqueda MMR
2. Knowledge Graph y consultas SPARQL
3. Los 5 casos de inferencia
4. Transformación de consultas (HyDE, Decomposition)
5. Agente completo con 3 casos de uso
6. Métricas y LLM como juez

---

## Flujo del Agente

```
Usuario → Query Transform → Retrieve → Generate → Reflect
               │                                      │
     (DIRECT/HYDE/DECOMPOSE)              ┌───────────┤
                                          │           │
                                    APROBADO → FIN   RECHAZADO
                                                      │
                                              retry < 3 → Retrieve (loop)
                                              retry = 3 → Web Fallback → Generate → FIN
```

| Nodo | Función |
|---|---|
| **Query Transform** | Clasifica la pregunta y aplica HyDE o Decomposition |
| **Retrieve** | Busca en ChromaDB (MMR) + Knowledge Graph (SPARQL) |
| **Generate** | Sintetiza respuesta con GPT-4o-mini usando el contexto |
| **Reflect** | LLM evalúa fidelidad, completitud y precisión legal |
| **Web Fallback** | DuckDuckGo si 3 reintentos fallan |

---

## Ontología OWL

Archivo: `ontology/insolvencia.ttl` — 356 triples

| Componente | Cantidad | Mínimo requerido |
|---|---|---|
| Clases (`owl:Class`) | 16 | 10 |
| Clases disjuntas | 2 pares | 2 |
| `rdfs:subClassOf` | 15+ | 3 |
| `rdfs:subPropertyOf` | 2 | 2 |
| Propiedades (object + datatype) | 18 | 10 |
| Individuos | 41 | 40 (4/clase) |
| `owl:inverseOf` | 2 | 2 |
| `owl:allValuesFrom` | 1 | 1 |
| `owl:someValuesFrom` | 1 | 1 |
| Restricción de cardinalidad | 1 | 1 |
| Constructor lógico (`owl:unionOf`) | 1 | 1 |

### Consultas SPARQL

Los archivos en `ontology/sparql/` contienen:

- **SELECT + ORDER BY + LIMIT**: Procedimientos ordenados por fecha, normas cronológicas
- **FILTER**: Procedimientos post-2023, normas con número "1116", acreedores "Banco"
- **UPDATE (INSERT DATA)**: Agregar procedimiento, vincular deudor
- **UPDATE (DELETE DATA)**: Eliminar individuo provisional

### 5 Casos de Inferencia

| # | Tipo | Resultado con GraphDB + Inferencia |
|---|---|---|
| 1 | SubClass (PersonaJuridica → Deudor) | 0 → 8 individuos |
| 2 | InverseOf (esAcreedorDe ← tieneAcreedor) | 0 → 6 triples |
| 3 | InverseOf (esIniciadoPor ← iniciaEn) | 0 → 3 triples |
| 4 | SubPropertyOf (tieneAcreedorPrivilegiado → tieneAcreedor) | 4 → 6 triples |
| 5 | EquivalentClass + UnionOf (Deudor ≡ PJ ∪ PN) | 0 → 8 individuos |

---

## Métricas de Evaluación

| Métrica | Descripción | Implementación |
|---|---|---|
| Recall@k | Fracción de docs relevantes en top-k | `src/evaluation/metrics.py` |
| Precision@k | Fracción de top-k que son relevantes | `src/evaluation/metrics.py` |
| MRR | Rank promedio del primer resultado relevante | `src/evaluation/metrics.py` |
| nDCG@k | Calidad del ranking ponderada por posición | `src/evaluation/metrics.py` |
| Faithfulness | ¿Respuesta basada en el contexto? | RAGAS wrapper |
| Answer Relevancy | ¿Respuesta aborda la pregunta? | RAGAS wrapper |
| LLM como Juez | Relevancia + Fidelidad + Precisión Legal (1-5) | `src/evaluation/llm_judge.py` |

---

## Stack Tecnológico

| Componente | Tecnología | Justificación |
|---|---|---|
| Lenguaje | Python 3.11+ | Requerido por la práctica |
| Orquestación | LangGraph | Soporta ciclos (retry loops), estado tipado |
| LLM | GPT-4o-mini | Balance costo/calidad para español legal |
| Embeddings | HuggingFace `paraphrase-multilingual-mpnet-base-v2` | Optimizado para español, gratuito, 768-dim |
| Vector Store | ChromaDB | MMR nativo, filtrado por metadata, persistencia local |
| Ontología | RDFLib + Turtle | Parsing OWL puro en Python |
| Knowledge Graph | GraphDB Free | SPARQL 1.1, razonamiento OWL, API REST |
| Trazabilidad | LangSmith | Integración automática con LangGraph |
| Web Fallback | DuckDuckGo | Sin API key, buen soporte de español |
| Evaluación | Numpy + RAGAS | Métricas estándar IR + evaluación RAG |

---

## Solución de Problemas

| Problema | Solución |
|---|---|
| `ModuleNotFoundError` | Activar venv: `source .venv/bin/activate` y `pip install -r requirements.txt` |
| `OPENAI_API_KEY not set` | Crear `.env` desde `.env.example` y agregar la clave |
| `Vector store not found` | Ejecutar `python main.py --ingest` primero |
| `GraphDB connection refused` | Normal si no está instalado. El sistema usa RDFLib local como fallback |
| Warnings `UNEXPECTED` | Ignorables. Son diferencias de versión en el modelo de embeddings |
| Respuesta cortada | El LLM puede truncar. Intente con una pregunta más específica |

---

## Enlace Video Sustentación

**[PENDIENTE — Insertar URL de YouTube aquí]**
