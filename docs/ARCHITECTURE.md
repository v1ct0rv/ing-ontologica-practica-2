# Knowledge Graph RAG System Architecture
## Colombian Business Insolvency Law Domain
### Práctica 2 — Ingeniería Ontológica

---

## Table of Contents

1. [High-Level Architecture Diagram](#1-high-level-architecture-diagram)
2. [Directory Structure](#2-directory-structure)
3. [Module Descriptions](#3-module-descriptions)
4. [Data Flow](#4-data-flow)
5. [LangGraph State Machine](#5-langgraph-state-machine)
6. [Technology Choices & Justification](#6-technology-choices--justification)
7. [Integration Points](#7-integration-points)
8. [Configuration](#8-configuration)
9. [Ontology Design Overview](#9-ontology-design-overview)
10. [Inference Cases (Part C.2 Requirement)](#10-inference-cases-part-c2-requirement)
11. [Deliverables](#11-deliverables)

---

## 1. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE (offline)                           │
│                                                                                 │
│  ┌──────────────┐    ┌────────────────────┐    ┌──────────────────────────┐    │
│  │  50 PDF docs │───▶│  SemanticChunker   │───▶│  HuggingFace             │    │
│  │ (content/    │    │  (LangChain)       │    │  paraphrase-multilingual │    │
│  │  docs/)      │    │                    │    │  -mpnet-base-v2 / 768-dim│    │
│  └──────────────┘    └────────────────────┘    └────────────┬─────────────┘    │
│                                                             │                  │
│  ┌──────────────┐    ┌────────────────────┐                 ▼                  │
│  │  ontology/   │───▶│  RDFLib parser     │───────▶ ┌──────────────┐          │
│  │  insolvency  │    │  (.ttl Turtle)     │         │   GraphDB    │          │
│  │  .ttl        │    └────────────────────┘         │  (SPARQL     │          │
│  └──────────────┘                                   │   endpoint)  │          │
│                                                     └──────────────┘          │
└─────────────────────────────────────────────────────┬───────────────────────── ┘
                                                      │
                                                      ▼
                                             ┌─────────────────┐
                                             │   ChromaDB      │
                                             │  (vector store) │
                                             │  MMR retrieval  │
                                             └────────┬────────┘
                                                      │
═══════════════════════════════════════════════════════════════════════════════════
                              QUERY-TIME PIPELINE
═══════════════════════════════════════════════════════════════════════════════════
                                                      │
┌─────────────┐                                       │
│    User     │                                       │
│   Query     │                                       │
└──────┬──────┘                                       │
       │                                              │
       ▼                                              │
┌──────────────────────────────────────────────────── │ ──────────────────────────┐
│                    LANGGRAPH AGENT GRAPH             │                           │
│                                                      │                           │
│  ┌─────────────────────┐                             │                           │
│  │   Query Analyzer    │◀────────────────────────────┘                           │
│  │  (decide strategy)  │                                                         │
│  │  · HyDE             │                                                         │
│  │  · Decomposition    │                                                         │
│  │  · Direct           │                                                         │
│  └──────────┬──────────┘                                                         │
│             │                                                                    │
│             ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐               │
│  │                     ReAct Agent (LangGraph)                  │               │
│  │                                                              │               │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │               │
│  │   │ vector_search│  │   kg_query   │  │   web_search     │  │               │
│  │   │  (ChromaDB   │  │  (SPARQL →   │  │  (Tavily/Google) │  │               │
│  │   │   MMR)       │  │   GraphDB)   │  │  [fallback only] │  │               │
│  │   └──────────────┘  └──────────────┘  └──────────────────┘  │               │
│  │                                                              │               │
│  │   ┌──────────────┐  ┌──────────────┐                         │               │
│  │   │  hyde_expand │  │  decompose   │                         │               │
│  │   │  (generate   │  │  _query      │                         │               │
│  │   │  hyp. doc)   │  │  (sub-quest) │                         │               │
│  │   └──────────────┘  └──────────────┘                         │               │
│  └──────────────────────────┬─────────────────────────────────── ┘               │
│                             │                                                    │
│                             ▼                                                    │
│                   ┌─────────────────────┐                                        │
│                   │  Response Generator │                                        │
│                   │  (GPT-4o synthesis) │                                        │
│                   └──────────┬──────────┘                                        │
│                              │                                                   │
│                              ▼                                                   │
│                   ┌─────────────────────┐         retry_count < 3               │
│                   │  Reflection / Critic│─── POOR ──────────────────────────┐   │
│                   │  (self-evaluation)  │                                    │   │
│                   └──────────┬──────────┘                                    │   │
│                              │ GOOD                   ┌─────────────────────┐│   │
│                              │                        │  Web Search Fallback││   │
│                              │        retry_count = 3 │  (Tavily forced)    ││   │
│                              │ ◀──── POOR ────────────└─────────────────────┘│   │
│                              │                        ▲                      │   │
│                              │                        └──────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│                   ┌─────────────────────┐                                        │
│                   │  Metrics Calculator │                                        │
│                   │  Recall@k, Prec@k   │                                        │
│                   │  MRR, nDCG, Faith.  │                                        │
│                   └──────────┬──────────┘                                        │
│                              │                                                   │
└──────────────────────────────│───────────────────────────────────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Final Response            │
              │   + Retrieved Sources           │
              │   + Evaluation Scores           │
              │   + LangSmith Trace URL         │
              └─────────────────────────────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
            ┌─────────────┐      ┌──────────────────┐
            │  Streamlit  │      │    LangSmith     │
            │   Web UI    │      │  (trace logs,    │
            │  (optional) │      │  eval dashboard) │
            └─────────────┘      └──────────────────┘
```

---

## 2. Directory Structure

```
Practica 2/
├── content/
│   └── docs/                        # 50 PDFs (already present)
├── ontology/
│   ├── insolvencia.ttl              # OWL ontology in Turtle
│   └── sparql/
│       ├── select_queries.sparql
│       ├── filter_queries.sparql
│       ├── update_queries.sparql
│       └── inference_cases.sparql
├── src/
│   ├── __init__.py
│   ├── config.py                    # Pydantic/dotenv settings
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py            # PyMuPDF-based loader
│   │   ├── semantic_chunker.py      # SemanticChunker wrapper
│   │   └── vector_store.py          # ChromaDB + MMR retriever
│   ├── kg/
│   │   ├── __init__.py
│   │   ├── ontology_manager.py      # RDFLib local graph
│   │   ├── graphdb_client.py        # SPARQLWrapper for GraphDB
│   │   └── sparql_tools.py          # LangChain tools for KG
│   ├── query/
│   │   ├── __init__.py
│   │   ├── hyde.py                  # HyDE implementation
│   │   ├── decomposer.py           # Query decomposition
│   │   └── router.py               # Query strategy classifier
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py                 # LangGraph AgentState
│   │   ├── tools.py                 # Tool definitions
│   │   ├── nodes.py                 # LangGraph node functions
│   │   ├── graph.py                 # StateGraph assembly
│   │   └── web_fallback.py          # Internet search fallback
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Recall@k, Precision@k, MRR, nDCG
│   │   └── llm_judge.py             # LLM-as-judge scoring
│   └── tracing/
│       ├── __init__.py
│       └── langsmith_setup.py       # LangSmith init
├── notebooks/
│   └── demo.ipynb
├── tests/
│   ├── test_ingestion.py
│   ├── test_kg.py
│   ├── test_agent.py
│   └── test_metrics.py
├── docs/
│   ├── REQUIREMENTS.md
│   ├── ARCHITECTURE.md
│   └── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## 3. Module Descriptions

### 3.1 `config/settings.py`

Uses `pydantic-settings` (`BaseSettings`) to load all configuration from environment variables and `.env` files. Provides a single typed `Settings` singleton imported everywhere else. Validates API keys at startup.

### 3.2 `ingestion/pdf_loader.py`

Wraps `pymupdf4llm` to load PDFs with layout-aware text extraction. Extracts document-level metadata (title, year, document type, legal norm reference) and attaches it as LangChain `Document` metadata. Handles encoding issues common in scanned Colombian legal PDFs.

### 3.3 `ingestion/chunker.py`

Wraps LangChain's `SemanticChunker` (from `langchain_experimental`). Configures breakpoint strategy (`percentile` with threshold 95 or `gradient`) appropriate for legal prose with long paragraphs. Produces `Document` chunks with inherited metadata plus `chunk_index` and `source_hash`.

### 3.4 `ingestion/vector_store.py`

Manages the ChromaDB `PersistentClient` and the HuggingFace embedding model. Creates/opens the `insolvency_chunks` collection. Provides a factory that returns a `MaxMarginalRelevanceRetriever` with configurable `k`, `fetch_k`, and `lambda_mult` (diversity parameter). Supports metadata filtering (e.g., filter by `doc_type=ley`). Reads source documents from `content/docs/`.

### 3.5 `kg/ontology_manager.py`

Parses `ontology/insolvencia.ttl` into an `rdflib.ConjunctiveGraph`. Provides an in-memory fallback if GraphDB is unavailable. Syncs the graph to GraphDB via the bulk import API on first run. Validates OWL consistency using `owlrl.DeductiveClosure` and verifies the 5 inference cases.

### 3.6 `kg/graphdb_client.py`

Thin HTTP client over GraphDB's REST SPARQL endpoint using `SPARQLWrapper`. Supports `SELECT`, `CONSTRUCT`, and `ASK` queries. Handles authentication, timeout, pagination (`LIMIT`/`OFFSET`), and formats results as typed Python dicts.

### 3.7 `kg/sparql_tools.py`

LangChain `Tool` definitions backed by SPARQL queries to GraphDB. Provides `kg_query` tool used by the ReAct agent. Accepts entity names and relationship predicates extracted from the user query, and produces well-formed SPARQL SELECT queries. Avoids SPARQL injection by using parameterized templates with namespace prefixes pre-declared.

### 3.8 `query/router.py`

LLM-based classifier that inspects the raw user query and returns one of three strategy tags:
- `HYDE`: single-concept factual question where a hypothetical answer improves retrieval
- `DECOMPOSE`: multi-part question requiring sub-queries
- `DIRECT`: narrow lookup, no transformation needed

Uses a fast, cheap model (GPT-4o-mini) with a structured output schema.

### 3.9 `query/hyde.py`

Implements Hypothetical Document Embeddings. Prompts the LLM to write a short hypothetical legal passage that would answer the query. Embeds the hypothetical document using `paraphrase-multilingual-mpnet-base-v2` and uses that 768-dim vector for ChromaDB retrieval instead of the raw query embedding. Falls back to direct query embedding on generation failure.

### 3.10 `query/decomposer.py`

Given a complex multi-part question, prompts the LLM to generate 2–5 atomic sub-questions. Each sub-question is processed independently through the retrieval pipeline, and results are merged and de-duplicated before synthesis.

### 3.13 `agent/state.py`

Defines `InsolvencyAgentState` as a `TypedDict` (LangGraph state schema):

```python
class InsolvencyAgentState(TypedDict):
    query: str
    strategy: Literal["HYDE", "DECOMPOSE", "DIRECT"]
    sub_queries: list[str]
    retrieved_chunks: list[Document]
    kg_context: dict
    web_results: list[dict]
    web_fallback_used: bool          # True after web_fallback_node fires; skips reflection
    response: str
    reflection_score: float
    reflection_feedback: str
    retry_count: int
    final_response: str
    metrics: dict
    trace_url: str
```

### 3.14 `agent/tools.py`

Defines LangChain `Tool` objects used by the ReAct agent:

| Tool Name | Description |
|---|---|
| `vector_search` | ChromaDB MMR retrieval, returns ranked chunks |
| `kg_query` | SPARQL query to GraphDB, returns structured graph context |
| `hyde_expand` | Generate hypothetical document, re-embed, retrieve |
| `decompose_query` | Break query into sub-questions, retrieve each |
| `web_search` | Tavily search (Colombian law sources preferred) |

### 3.15 `agent/nodes.py`

Pure functions implementing each LangGraph node. Each node takes `InsolvencyAgentState` and returns a partial state update dict:

- `query_analyzer_node`: calls `analyzer.py`, sets `strategy`
- `react_agent_node`: runs the ReAct loop with available tools
- `response_generator_node`: synthesizes retrieved evidence into a final answer
- `reflection_node`: scores response on faithfulness, relevance, completeness (0–1)
- `web_fallback_node`: forces Tavily search when retries are exhausted
- `metrics_node`: computes Recall@k, Precision@k, MRR, nDCG, calls RAGAS

### 3.16 `agent/edges.py`

Conditional edge functions:

- `route_after_response_generator`: if `web_fallback_used == True` → `metrics_node` (skip reflection to avoid infinite loop); else → `reflection_node`
- `route_after_reflection`: if `reflection_score >= 0.75` → `metrics_node`; else if `retry_count < 3` → `react_agent_node`; else → `web_fallback_node`
- `route_after_strategy`: dispatches to HyDE branch, decomposition branch, or direct tool call in ReAct

### 3.17 `agent/graph.py`

Assembles and compiles the `StateGraph`:

```python
graph = StateGraph(InsolvencyAgentState)
graph.add_node("query_analyzer", query_analyzer_node)
graph.add_node("react_agent", react_agent_node)
graph.add_node("response_generator", response_generator_node)
graph.add_node("reflection", reflection_node)
graph.add_node("web_fallback", web_fallback_node)
graph.add_node("metrics", metrics_node)
graph.set_entry_point("query_analyzer")
graph.add_edge("query_analyzer", "react_agent")
graph.add_edge("react_agent", "response_generator")
# After response_generator: skip reflection if web fallback was used (avoids infinite loop)
graph.add_conditional_edges("response_generator", route_after_response_generator)
graph.add_conditional_edges("reflection", route_after_reflection)
# web_fallback sets web_fallback_used=True, then goes to response_generator for a single synthesis pass
graph.add_edge("web_fallback", "response_generator")
graph.add_edge("metrics", END)
app = graph.compile(checkpointer=MemorySaver())
```

### 3.18 `evaluation/metrics.py`

Implements retrieval metrics from scratch using only numpy:
- `recall_at_k(retrieved_ids, relevant_ids, k)`
- `precision_at_k(retrieved_ids, relevant_ids, k)`
- `mrr(retrieved_ids, relevant_ids)`
- `ndcg_at_k(retrieved_ids, relevant_scores, k)` — uses graded relevance

### 3.19 `evaluation/ragas_evaluator.py`

Wraps the `ragas` library to compute:
- **Faithfulness**: are all answer claims grounded in the retrieved context?
- **Answer Relevance**: does the answer address the actual question?
- **Context Precision**: are the retrieved chunks actually relevant?
- **Context Recall**: does the context cover what's needed?

Converts LangChain `Document` objects to RAGAS `Dataset` format.

### 3.20 `evaluation/llm_judge.py`

Implements LLM-as-judge using GPT-4o with a structured JSON rubric. Scores responses on:
- Legal accuracy (Colombian insolvency law correctness)
- Completeness
- Citation quality (does it reference specific articles/norms?)
- Clarity

Returns scores 1–5 per dimension with justification text.

### 3.21 `tracing/langsmith_setup.py`

Initializes LangSmith `Client`, sets up a named project (`insolvency-kg-rag`), and provides a decorator/context manager to wrap agent runs. Attaches metadata tags (query strategy, retry count, model versions) to each trace.

---

## 4. Data Flow

### 4.1 Offline Ingestion (run once)

```
Step 1: PDF Loading
  content/docs/*.pdf
    → pymupdf4llm.to_markdown()
    → LangChain Document(page_content, metadata={source, doc_type, year, norm_ref})

Step 2: Semantic Chunking
  Document list
    → SemanticChunker(embeddings=HuggingFaceEmbeddings(model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"), breakpoint_threshold_type="percentile")
    → List[Document] (avg ~400 tokens/chunk, semantically coherent)

Step 3: Embedding + Indexing
  List[Document]
    → HuggingFaceEmbeddings(model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")  # 768-dim vectors, optimized for Spanish legal text
    → Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

Step 4: Ontology Loading
  ontology/insolvencia.ttl
    → rdflib.Graph().parse(format="turtle")
    → Validate with owlrl.DeductiveClosure(owlrl.OWLRL_Semantics)
    → POST to GraphDB bulk import endpoint
    → Confirm with SPARQL ASK { ?s ?p ?o } COUNT
```

### 4.2 Query-Time Pipeline

```
Step 1: User submits query (UI or API)
  "¿Cuáles son los requisitos para iniciar un proceso de reorganización
   empresarial bajo la Ley 1116 de 2006?"

Step 2: LangSmith trace begins (run_id assigned)

Step 3: query_analyzer_node
  → LLM classifies → strategy = "DIRECT" (single focused question)
  → State updated: {strategy: "DIRECT"}

Step 4: react_agent_node (ReAct loop)
  Thought: "I need to find legal requirements for reorganización empresarial"
  Action: vector_search(query="requisitos reorganización empresarial Ley 1116")
    → ChromaDB MMR returns top-8 chunks (k=8, fetch_k=20, lambda_mult=0.7)
  Observation: [chunk_1: "Art. 9 Ley 1116: Para acceder al proceso...", ...]

  Thought: "I should verify entities and relationships in the knowledge graph"
  Action: kg_query(entity="ReorganizacionEmpresarial", relation="tieneRequisito")
    → SPARQL SELECT → GraphDB returns {requisitos: [cesacionPagos, solicitudEscrita, ...]}
  Observation: {entity: "ReorganizacionEmpresarial", requisitos: [...]}

  Thought: "Sufficient context gathered"
  → Stop ReAct loop, pass evidence to response_generator

Step 5: response_generator_node
  → Synthesizes vector chunks + KG triples into coherent answer
  → GPT-4o with system prompt enforcing legal citation style
  → Draft response with article references

Step 6: reflection_node
  → Critic LLM evaluates draft:
      faithfulness: 0.92
      relevance: 0.88
      completeness: 0.85
      composite_score: 0.88  → GOOD (≥ 0.75)
  → Route to metrics_node

Step 7: metrics_node
  → Recall@5: 0.80, Precision@5: 0.75
  → MRR: 0.83, nDCG@5: 0.79
  → RAGAS faithfulness: 0.91
  → LLM-judge legal accuracy: 4.5/5

Step 8: Final response returned
  → answer text + source citations + metric scores + LangSmith trace URL
```

### 4.3 Retry Flow (poor response)

```
reflection_node → score = 0.52 (POOR), retry_count = 0
  → retry_count++ = 1
  → reflect feedback: "Missing reference to Art. 9 cesación de pagos definition"
  → back to react_agent_node

react_agent_node (retry 1)
  → Uses reflection_feedback as additional context in ReAct prompt
  → Adds decompose_query tool call to widen coverage
  → New evidence gathered

response_generator_node → reflection_node → score = 0.71 (POOR), retry_count = 1
  → retry_count++ = 2
  → back to react_agent_node (retry 2)

response_generator_node → reflection_node → score = 0.82 (GOOD), retry_count = 2
  → Route to metrics_node → Final response
```

### 4.4 Web Fallback Flow

```
reflection_node → score = 0.48 (POOR), retry_count = 3
  → Route to web_fallback_node

web_fallback_node
  → Tavily search: "reorganización empresarial Ley 1116 Colombia requisitos"
  → Returns top-5 web results (preferring .gov.co, .supersociedades.gov.co)
  → Appended to state.web_results
  → Sets state.web_fallback_used = True

response_generator_node
  → Synthesizes from original context + web results
  → Marked in response metadata: sources_include_web=True
  → route_after_response_generator detects web_fallback_used=True
  → reflection_node SKIPPED (prevents infinite loop: response_generator → reflection → web_fallback → ...)
  → directly to metrics_node
```

---

## 5. LangGraph State Machine

### 5.1 Nodes and Edges

```
                    ┌─────────────────┐
                    │   START (entry) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  query_analyzer │  (classify strategy)
                    └────────┬────────┘
                             │ always
                             ▼
                    ┌─────────────────┐
                    │   react_agent   │  (tool-use loop)
                    └────────┬────────┘
                             │ always
                             ▼
                    ┌─────────────────────┐
                    │  response_generator │  (synthesize answer)
                    └──────────┬──────────┘
                               │ always
                               ▼
                    ┌─────────────────┐
                    │   reflection    │  (score 0–1)
                    └────────┬────────┘
                             │
              ┌──────────────┼───────────────────┐
              │              │                   │
    score≥0.75│    score<0.75│                   │score<0.75
    (GOOD)    │  retry<3     │                   │retry=3
              ▼              ▼                   ▼
         ┌─────────┐  ┌─────────────┐   ┌──────────────────┐
         │ metrics │  │ react_agent │   │  web_fallback    │
         └────┬────┘  │  (retry++)  │   │  (Tavily forced) │
              │       └─────────────┘   └────────┬─────────┘
              │                                  │ always
              │                                  ▼
              │                       ┌─────────────────────┐
              │                       │  response_generator │
              │                       └──────────┬──────────┘
              │                                  │ always
              │                                  ▼
              │                       ┌─────────────────┐
              │                       │     metrics     │
              │                       └────────┬────────┘
              │                                │
              └────────────────────────────────┘
                                               │
                                               ▼
                                          ┌─────────┐
                                          │   END   │
                                          └─────────┘
```

### 5.2 State Transitions Table

| From Node | Condition | To Node |
|---|---|---|
| START | always | `query_analyzer` |
| `query_analyzer` | always | `react_agent` |
| `react_agent` | always | `response_generator` |
| `response_generator` | `web_fallback_used == False` | `reflection` |
| `response_generator` | `web_fallback_used == True` | `metrics` |
| `reflection` | `score >= 0.75` | `metrics` |
| `reflection` | `score < 0.75 AND retry_count < 3` | `react_agent` |
| `reflection` | `score < 0.75 AND retry_count == 3` | `web_fallback` |
| `web_fallback` | always | `response_generator` (sets `web_fallback_used=True`) |
| `metrics` | always | END |

### 5.3 ReAct Agent Inner Loop

The `react_agent` node itself uses LangChain's `create_react_agent` with the following tool-selection logic embedded in the system prompt:

```
TOOLS AVAILABLE:
1. vector_search     → use for: finding relevant legal passages, article text, doctrine
2. kg_query          → use for: entity relationships, procedure steps, norm taxonomy
3. hyde_expand       → use for: vague queries where hypothetical doc improves retrieval
4. decompose_query   → use for: multi-part questions (≥2 distinct legal concepts)
5. web_search        → use ONLY when explicitly instructed by the fallback mechanism

STRATEGY GUIDANCE (injected from query_analyzer result):
- If strategy=HYDE: first action MUST be hyde_expand
- If strategy=DECOMPOSE: first action MUST be decompose_query
- If strategy=DIRECT: choose between vector_search and kg_query based on question type
```

---

## 6. Technology Choices & Justification

### 6.1 Language & Runtime

| Choice | Version | Justification |
|---|---|---|
| Python | 3.11+ | `match` statements for LangGraph edges; `tomllib` stdlib; performance improvements for async |
| `pyproject.toml` | PEP 621 | Modern packaging, replaces `setup.py` |

### 6.2 Orchestration

| Choice | Version | Justification |
|---|---|---|
| LangChain | 0.3.x | Mature retriever abstractions, SemanticChunker, tool definitions |
| LangGraph | 0.2.x | Cycles (retry loops) not possible in plain LangChain; typed state; MemorySaver for session continuity |
| LangSmith | 0.2.x | First-class LangChain integration; auto-traces all LLM calls without instrumentation boilerplate |

### 6.3 LLM & Embeddings

| Choice | Model | Justification |
|---|---|---|
| OpenAI | `gpt-4o` | Best legal reasoning; structured outputs for reflection scoring |
| OpenAI | `gpt-4o-mini` | Query classification, fast/cheap; cost-effective for high-frequency calls |
| HuggingFace | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768-dim; purpose-built for multilingual semantic similarity; no API key required; superior Spanish legal text performance vs. OpenAI embeddings |

Note: Embeddings run locally via `langchain-huggingface` (`HuggingFaceEmbeddings`). No OpenAI API key is needed for the ingestion or retrieval pipeline — only for LLM generation. Anthropic Claude 3.5 Sonnet is a supported alternative for the main synthesis LLM via `langchain_anthropic.ChatAnthropic`.

### 6.4 Vector Store

| Choice | Version | Justification |
|---|---|---|
| ChromaDB | 0.5.x | Persistent local storage with no external service required during development; native MMR support; metadata filtering for doc_type/year |
| FAISS | (alt.) | Better raw throughput for very large corpora; no metadata filtering without wrapper |

ChromaDB is preferred here because the 50-document corpus (~5,000–15,000 chunks) fits comfortably in local storage, and metadata filtering by legal norm type is needed.

### 6.5 Knowledge Graph

| Choice | Version | Justification |
|---|---|---|
| RDFLib | 7.x | Pure Python OWL/Turtle parsing; in-memory SPARQL fallback for tests |
| GraphDB Free | 10.x | Production-grade SPARQL 1.1 endpoint; OWL reasoning; free tier supports corpus size; REST API for easy integration |
| owlrl | 0.9.x | OWL RL closure computation to infer implicit triples at load time |

### 6.6 Document Processing

| Choice | Version | Justification |
|---|---|---|
| pymupdf4llm | 0.0.17+ | Layout-aware PDF → Markdown; handles two-column legal documents better than `pypdf` |
| langchain_experimental | 0.3.x | Contains `SemanticChunker` |

### 6.7 Evaluation

| Choice | Version | Justification |
|---|---|---|
| RAGAS | 0.2.x | Industry standard for RAG evaluation; Faithfulness and Context Recall metrics are domain-agnostic |
| numpy | 2.x | Custom Recall@k, MRR, nDCG implementations for full control |

### 6.8 Web Search (Fallback)

| Choice | Version | Justification |
|---|---|---|
| Tavily | latest | LangChain-native integration; supports domain-restricted search (`.gov.co`); returns structured snippets |

### 6.9 UI

| Choice | Version | Justification |
|---|---|---|
| Streamlit | 1.35+ | Rapid prototyping; built-in chat components (`st.chat_message`); easy LangGraph integration via `asyncio` |

---

## 7. Integration Points

### 7.1 Vector Store + LLM (HyDE)

```
User query (raw)
    │
    ▼ hyde.py
OpenAI gpt-4o-mini: "Write a Colombian legal passage that answers: [query]"
    │
    ▼
Hypothetical document text
    │
    ▼ ingestion/vector_store.py
paraphrase-multilingual-mpnet-base-v2 → 768-dim vector
    │
    ▼ chroma_store.py
ChromaDB.query(embedding=hyp_vector, n_results=20) → MMR(k=8, lambda=0.7)
    │
    ▼
Top-8 semantically diverse, relevant chunks
```

### 7.2 Knowledge Graph + Vector Store (Hybrid Retrieval)

The agent combines both sources by interleaving tool calls:

1. `vector_search` retrieves lexically/semantically similar passages
2. `kg_query` retrieves structured graph context (entity definitions, norm hierarchy, procedure steps)
3. `response_generator_node` merges both into a single context window:
   - Vector chunks: provide verbatim article text and doctrine
   - KG triples: provide structured relationships that guide synthesis ("ReorganizacionEmpresarial `reglamentadoPor` Ley1116Art9")

This hybrid approach prevents hallucinations about entity relationships while leveraging the vector store for verbatim legal text.

### 7.3 LangGraph + LangSmith

All LangGraph node invocations are automatically traced by LangSmith when `LANGCHAIN_TRACING_V2=true` is set. Each node call appears as a child span in the trace tree. Additional metadata is attached via `langsmith.traceable` decorator on the tool functions, capturing:

- Tool name and inputs
- Retrieved chunk IDs and scores
- SPARQL query text
- Reflection scores and feedback
- Retry count at each reflection

### 7.4 Reflection + State Accumulation

The `retry_count` field in `InsolvencyAgentState` is incremented by `reflection_node` before routing back to `react_agent`. The `reflection_feedback` string (e.g., "Answer lacks citation of Art. 9 on cesación de pagos") is injected into the ReAct system prompt on the next iteration, giving the agent targeted guidance without resetting accumulated retrieved chunks. The `retrieved_chunks` list is append-only across retries.

---

## 8. Configuration

### 8.1 Required Environment Variables (`.env`)

```bash
# LLM Provider (used for generation only)
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_FAST_MODEL=gpt-4o-mini

# Embeddings (HuggingFace — no API key required)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
EMBEDDING_DEVICE=cpu                  # or "cuda" if GPU available

# Optional: Anthropic fallback
ANTHROPIC_API_KEY=sk-ant-...

# LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=insolvency-kg-rag

# Vector Store
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=insolvency_chunks

# GraphDB
GRAPHDB_ENDPOINT=http://localhost:7200
GRAPHDB_REPOSITORY=insolvency
GRAPHDB_USERNAME=admin
GRAPHDB_PASSWORD=

# Web Search Fallback
TAVILY_API_KEY=tvly-...

# Evaluation
RAGAS_OPENAI_MODEL=gpt-4o-mini
LLM_JUDGE_MODEL=gpt-4o

# Ingestion
CORPUS_DIR=./content/docs
CHUNK_BREAKPOINT_TYPE=percentile
CHUNK_BREAKPOINT_THRESHOLD=95
MMR_K=8
MMR_FETCH_K=20
MMR_LAMBDA_MULT=0.7

# Agent
MAX_REFLECTION_RETRIES=3
REFLECTION_PASS_THRESHOLD=0.75
REACT_MAX_ITERATIONS=8
```

### 8.2 `pyproject.toml` Dependencies

```toml
[project]
name = "insolvency-kg-rag"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-anthropic>=0.2.0",
    "langchain-community>=0.3.0",
    "langchain-experimental>=0.3.0",
    "langgraph>=0.2.0",
    "langsmith>=0.2.0",
    "chromadb>=0.5.0",
    "openai>=1.40.0",
    "anthropic>=0.34.0",
    "rdflib>=7.0.0",
    "owlrl>=0.9.0",
    "pymupdf4llm>=0.0.17",
    "httpx>=0.27.0",
    "ragas>=0.2.0",
    "numpy>=2.0.0",
    "pydantic-settings>=2.4.0",
    "tavily-python>=0.5.0",
    "streamlit>=1.35.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.12.0",
]
```

### 8.3 GraphDB Setup

```bash
# Start GraphDB (Docker)
docker run -d \
  --name graphdb \
  -p 7200:7200 \
  -v $(pwd)/graphdb_data:/opt/graphdb/home \
  ontotext/graphdb:10.7.3

# Create repository via REST
curl -X POST http://localhost:7200/rest/repositories \
  -H "Content-Type: multipart/form-data" \
  -F "config=@graphdb_config.ttl"
```

Where `graphdb_config.ttl` configures the repository with OWL-RL reasoning enabled.

---

## 9. Ontology Design Overview

### 9.1 Namespace Declarations

```turtle
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix ins:  <http://www.unal.edu.co/ontologies/insolvencia#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
```

### 9.2 Core Classes

```
ins:EntidadJuridica            (Legal Entity — top-level)
  ├── ins:Deudor               (Debtor)
  │    ├── ins:PersonaJuridica  (Legal Person / Company)
  │    └── ins:PersonaNatural   (Natural Person / Individual)
  ├── ins:Acreedor             (Creditor)
  │    ├── ins:AcreedorPrivilegiado  (Secured / Privileged Creditor)
  │    └── ins:AcreedorOrdinario     (Unsecured / Ordinary Creditor)
  └── ins:Liquidador           (Court-appointed Liquidator)

ins:ProcedimientoInsolvencia   (Insolvency Procedure — top-level)
  ├── ins:Reorganizacion       (Reorganization Process — Art. 5 Ley 1116)
  └── ins:Liquidacion          (Judicial Liquidation — Art. 49 Ley 1116)

ins:ActoJuridico               (Legal / Procedural Act)
  ├── ins:AutoAdmision         (Admission Order issued by Superintendencia)
  └── ins:AcuerdoReorganizacion (Reorganization Agreement)

ins:Obligacion                 (Financial Obligation / Debt)

ins:Garantia                   (Security Interest / Collateral)

ins:OrganoDecisorio            (Decision-making Body)
  ├── ins:JuntaAcreedores      (Creditors' Assembly)
  └── ins:Superintendencia     (Superintendencia de Sociedades — supervising authority)

ins:NormaLegal                 (Legal Norm)
  ├── ins:Ley                  (Statutory Law)
  └── ins:Decreto              (Executive Decree)
```

### 9.3 Key Object Properties

```turtle
ins:tieneAcreedor        rdfs:domain ins:ProcedimientoInsolvencia ;
                         rdfs:range  ins:Acreedor .
                         # Links a procedure to its creditors

ins:esAcreedorDe         owl:inverseOf ins:tieneAcreedor .
                         # Inverse: Acreedor → ProcedimientoInsolvencia

ins:iniciaEn             rdfs:domain ins:ProcedimientoInsolvencia ;
                         rdfs:range  ins:ActoJuridico .
                         # First procedural act triggering the process (e.g., AutoAdmision)

ins:esSupervisadoPor     rdfs:domain ins:ProcedimientoInsolvencia ;
                         rdfs:range  ins:OrganoDecisorio .
                         # Supervising authority (Superintendencia) or decision body

ins:involucraDeudor      rdfs:domain ins:ProcedimientoInsolvencia ;
                         rdfs:range  ins:Deudor .

ins:tieneObligacion      rdfs:domain ins:Deudor ;
                         rdfs:range  ins:Obligacion .

ins:garantizadaCon       rdfs:domain ins:Obligacion ;
                         rdfs:range  ins:Garantia .

ins:reguladaPor          rdfs:domain ins:ProcedimientoInsolvencia ;
                         rdfs:range  ins:NormaLegal .

ins:precedeA             rdfs:domain ins:ActoJuridico ;
                         rdfs:range  ins:ActoJuridico .
                         # Transitive — models sequencing of procedural acts
```

### 9.4 Key Datatype Properties

```turtle
ins:numeroNorma          rdfs:domain ins:NormaLegal ;
                         rdfs:range  xsd:string .
                         # e.g., "1116" for Ley 1116

ins:añoExpedicion        rdfs:domain ins:NormaLegal ;
                         rdfs:range  xsd:gYear .

ins:nombreOficial        rdfs:domain ins:NormaLegal ;
                         rdfs:range  xsd:string .

ins:montoObligacion      rdfs:domain ins:Obligacion ;
                         rdfs:range  xsd:decimal .

ins:descripcion          rdfs:domain owl:Thing ;
                         rdfs:range  xsd:string .
                         # rdfs:subPropertyOf skos:definition — general description/label property

ins:fechaInicio          rdfs:domain ins:ProcedimientoInsolvencia ;
                         rdfs:range  xsd:date .
```

### 9.5 Representative Triples (Turtle Snippet)

```turtle
ins:ProcReorganizacion_EmpresaX
    a ins:Reorganizacion ;
    rdfs:label "Proceso de Reorganización — Empresa X"@es ;
    ins:reguladaPor ins:Ley1116 ;
    ins:esSupervisadoPor ins:Superintendencia ;
    ins:involucraDeudor ins:EmpresaX ;
    ins:tieneAcreedor ins:BancoColombia ;
    ins:tieneAcreedor ins:AcreedorOrdinario_001 ;
    ins:iniciaEn ins:AutoAdmision_EmpresaX .

ins:EmpresaX
    a ins:PersonaJuridica ;
    rdfs:label "Empresa X S.A.S."@es ;
    ins:descripcion "Sociedad por acciones simplificada en cesación de pagos."@es .

ins:BancoColombia
    a ins:AcreedorPrivilegiado ;
    rdfs:label "Banco de Colombia"@es .

ins:AutoAdmision_EmpresaX
    a ins:AutoAdmision ;
    rdfs:label "Auto de Admisión — Empresa X"@es ;
    ins:precedeA ins:AcuerdoReorganizacion_EmpresaX .

ins:AcuerdoReorganizacion_EmpresaX
    a ins:AcuerdoReorganizacion ;
    rdfs:label "Acuerdo de Reorganización — Empresa X"@es .

ins:Ley1116
    a ins:Ley ;
    ins:nombreOficial "Ley 1116 de 2006"@es ;
    ins:numeroNorma "1116" ;
    ins:añoExpedicion "2006"^^xsd:gYear ;
    rdfs:label "Régimen de Insolvencia Empresarial"@es .
```

### 9.6 SPARQL Example Queries

**Get all creditors involved in a reorganization procedure:**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?procedimiento ?acreedor ?label WHERE {
    ?procedimiento a ins:Reorganizacion ;
                   ins:tieneAcreedor ?acreedor .
    OPTIONAL { ?acreedor rdfs:label ?label }
}
```

**Get the supervising authority and initial act for all active procedures:**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?procedimiento ?organo ?actoInicial WHERE {
    ?procedimiento a ins:ProcedimientoInsolvencia ;
                   ins:esSupervisadoPor ?organo ;
                   ins:iniciaEn ?actoInicial .
    OPTIONAL { ?procedimiento rdfs:label ?label }
}
ORDER BY ?procedimiento
```

**Find all norms regulating insolvency procedures:**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?procedimiento ?norma ?nombreNorma WHERE {
    ?procedimiento a ins:ProcedimientoInsolvencia ;
                   ins:reguladaPor ?norma .
    OPTIONAL { ?norma ins:nombreOficial ?nombreNorma }
    OPTIONAL { ?procedimiento rdfs:label ?procLabel }
}
```

---

## 10. Inference Cases (Part C.2 Requirement)

The ontology must define at least **5 inference cases** using OWL-RL semantics, exercised through the SPARQL file `ontology/sparql/inference_cases.sparql`. The cases to implement are:

| # | Inference Type | Description |
|---|---|---|
| 1 | **Inverse property** | If `ins:tieneAcreedor(P, A)` then `ins:esAcreedorDe(A, P)` (via `owl:inverseOf`) |
| 2 | **Transitive property** | If `AutoAdmision ins:precedeA AcuerdoReorganizacion` and `AcuerdoReorganizacion ins:precedeA Liquidacion` then `AutoAdmision ins:precedeA Liquidacion` (via `owl:TransitiveProperty`) |
| 3 | **Subclass inheritance** | A `ins:PersonaJuridica` is inferred to be both a `ins:Deudor` and an `ins:EntidadJuridica` through `rdfs:subClassOf` chain |
| 4 | **Domain/Range entailment** | If `X ins:esSupervisadoPor Y`, OWL-RL infers `X rdf:type ins:ProcedimientoInsolvencia` and `Y rdf:type ins:OrganoDecisorio` from domain/range axioms |
| 5 | **Equivalent class** | `ins:Reorganizacion owl:equivalentClass [ owl:intersectionOf (ins:ProcedimientoInsolvencia [ owl:hasValue ins:Superintendencia ]) ]` — allows reasoner to classify individuals automatically |

These cases are validated at ontology-load time via `owlrl.DeductiveClosure(owlrl.OWLRL_Semantics)` and verified with SPARQL ASK queries in `inference_cases.sparql`.

---

## 11. Deliverables

### 11.1 Code & Artifacts

| Deliverable | Location |
|---|---|
| OWL Ontology (Turtle) | `ontology/insolvencia.ttl` |
| SPARQL Queries (5 types) | `ontology/sparql/` |
| RAG Agent implementation | `src/` |
| Evaluation report | run output / LangSmith dashboard |
| Demo notebook | `notebooks/demo.ipynb` |

### 11.2 YouTube Video Deliverable

A short video demonstration (approximately 5–10 minutes) must be published to YouTube as part of the Práctica 2 submission. The video should cover:

1. **Ontology walkthrough** — show the class hierarchy, properties, and 5 inference cases in Protégé or GraphDB
2. **Ingestion pipeline** — demonstrate loading the 50 PDFs into ChromaDB using the HuggingFace multilingual embeddings
3. **Agent query demo** — run at least 3 example queries through the LangGraph agent, showing the ReAct loop, reflection, and (optionally) web fallback
4. **Evaluation results** — display Recall@k, MRR, nDCG, and LLM-judge scores from the evaluation suite
5. **LangSmith trace** — walk through one complete trace in the LangSmith dashboard

The YouTube URL must be included in the final submission document.

---

*Architecture version 1.1 — April 2026*
*Domain: Colombian Business Insolvency Law*
*Stack: Python 3.11 · LangChain 0.3 · LangGraph 0.2 · ChromaDB 0.5 · GraphDB 10 · RDFLib 7 · RAGAS 0.2 · sentence-transformers*
