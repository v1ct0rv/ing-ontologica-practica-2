# Práctica 2: Knowledge Graph RAG System
## Ingeniería Ontológica — Universidad Nacional de Colombia

**Date:** March 2026
**Domain:** Colombian Business Insolvency Law (Ley 1116 de 2006, Decreto 806 de 2020)
**Technology Stack:** Python, LangGraph, LangSmith, GraphDB, RDFLib, FAISS/Chroma

---

## TABLE OF CONTENTS
1. [Domain Overview](#domain-overview)
2. [Part A: Agentic RAG Avanzado](#part-a--agentic-rag-avanzado)
3. [Part B: System Flow](#part-b--general-system-flow)
4. [Part C: Knowledge Graph](#part-c--knowledge-graph)
5. [Part D: LangSmith Traceability](#part-d--langsmith-traceability)
6. [Deliverables](#deliverables)
7. [Evaluation Rubric](#evaluation-rubric)

---

## DOMAIN OVERVIEW

### Content Specification
- **Total Documents:** 50 PDF documents (minimum)
- **Location:** `content/docs/` directory
- **Document Types:** PDF, TXT, HTML (primary focus: PDFs)
- **Subject Matter:**
  - Ley 1116 de 2006 (Colombian Insolvency Law)
  - Decreto 806 de 2020
  - Reorganization procedures
  - Legal concepts on cessation of payments (cese de pagos)
  - Academic doctrine on business insolvency
  - Jurisprudential analyses and case law

### Document Status
- [x] 50 documents present in `content/docs/`

---

## PART A — AGENTIC RAG AVANZADO

### 1. Corpus Construction

#### Requirements
| Requirement | Minimum | Status | Notes |
|------------|---------|--------|-------|
| Number of documents | 50 | [ ] | PDF/TXT/HTML formats |
| Domain coverage | Representative | [ ] | Must span multiple aspects of Colombian insolvency law |
| Ontology design | Basic OWL | [ ] | See Part C for detailed specification |

#### Checklist
- [ ] Verify all 50 documents are properly accessible in `content/docs/`
- [ ] Create document inventory with titles and word counts
- [ ] Categorize documents by legal topic (law texts, regulations, case analyses, doctrine)
- [ ] Prepare documents for ingestion (format validation, OCR if needed)

---

### 2. Ingestion, Indexing & Semantic Chunking

#### Requirements
| Requirement | Specification | Status |
|------------|---------------|--------|
| Document loading | PDF/TXT/HTML parsing | [ ] |
| Semantic chunking | Non-fixed-size chunks | [ ] |
| Chunking library | SemanticChunker | [ ] |
| Vector store | FAISS, Chroma, or equivalent | [ ] |
| Vector embeddings | LLM-based embeddings | [ ] |

#### Technical Details

**Semantic Chunking Implementation:**
- Use `SemanticChunker` to split documents based on semantic meaning (not fixed token windows)
- Preserve legal concepts and paragraph structure
- Overlap strategy: retain context without redundancy
- Handle tables, lists, and legal definitions appropriately

**Vector Store Configuration:**
- Index type: Dense vector index (FAISS recommended for performance)
- Embedding dimension: 1536 (OpenAI) or equivalent
- Metadata stored with chunks: document source, page number, section title, chunk ID
- Total chunks expected: ~5,000-10,000 (depending on document size)

#### Checklist
- [ ] Implement document loader for PDF (PyPDF2/pdfplumber)
- [ ] Implement document loader for TXT
- [ ] Implement document loader for HTML
- [ ] Configure SemanticChunker with appropriate semantic threshold
- [ ] Create vector store instance (FAISS or Chroma)
- [ ] Implement batch ingestion pipeline with progress tracking
- [ ] Validate chunk quality (check semantic coherence of 10 random chunks)
- [ ] Store index metadata (chunk count, vector dimensions, embedding model)

---

### 3. Query Processing & Transformation

#### A. HyDE (Hypothetical Document Embeddings)

| Feature | Specification | Status |
|---------|---------------|--------|
| Purpose | Generate hypothetical documents for ambiguous/short queries | [ ] |
| Use case | Improve retrieval precision for vague questions | [ ] |
| Implementation | Prompt LLM to generate N hypothetical documents | [ ] |

**Example Workflow:**
- User Query: "What happens when a company stops paying?"
- HyDE Output: Generate 2-3 hypothetical legal documents about cessation of payments
- Embed hypothetical docs → Search vector store → Return top relevant documents

#### B. Query Decomposition

| Feature | Specification | Status |
|---------|---------------|--------|
| Purpose | Break multi-part questions into sub-queries | [ ] |
| Detection | Identify conjunctions, multiple topics | [ ] |
| Decomposition | Sequential or parallel sub-query execution | [ ] |
| Aggregation | Combine results for final answer | [ ] |

**Example Workflow:**
- Complex Query: "What is the definition of ceasing payments under Ley 1116, and what are the legal consequences?"
- Decomposed Queries:
  1. "Definition of ceasing payments (cese de pagos) in Ley 1116"
  2. "Legal consequences of ceasing payments"
- Execute both queries → Aggregate results

#### Checklist
- [ ] Implement HyDE prompt template for Colombian insolvency law domain
- [ ] Test HyDE with 5 sample ambiguous queries
- [ ] Implement query decomposition detector (rule-based or LLM)
- [ ] Create sub-query generation prompt
- [ ] Implement sequential execution strategy
- [ ] Implement parallel execution strategy with result merging
- [ ] Document decomposition rules for multi-part queries

---

### 4. Agent Architecture (ReAct + Reflecting)

#### A. ReAct Agent (Reasoning + Acting)

| Component | Requirement | Status |
|-----------|-------------|--------|
| Orchestration | LangGraph state machine | [ ] |
| Reasoning | Think step-by-step (CoT) | [ ] |
| Tool selection | Dynamic tool calling | [ ] |
| Tool list | See below | [ ] |

**Available Tools for ReAct Agent:**
1. **Vector Search Tool** — Retrieve semantically similar documents from FAISS/Chroma
2. **Knowledge Graph Query Tool** — Execute SPARQL queries on GraphDB
3. **HyDE Transform Tool** — Generate hypothetical documents for query enhancement
4. **Query Decomposition Tool** — Break down complex questions
5. **Internet Search Tool** — Search web (used after 3 failed attempts)
6. **Evidence Synthesis Tool** — Combine multiple sources into coherent response

**ReAct Workflow:**
```
Thought → Action (select tool) → Observation (tool result)
→ Thought → ... → Final Answer
```

#### B. Reflecting Agent (Self-Evaluation & Retry)

| Feature | Specification | Status |
|---------|---------------|--------|
| Preliminary answer | Generated from retrieval | [ ] |
| Self-evaluation | LLM critiques own answer | [ ] |
| Criticism logic | Checks completeness, accuracy, citation | [ ] |
| Retry limit | Max 3 attempts | [ ] |
| Success criteria | Sufficient citation + completeness | [ ] |

**Reflecting Workflow:**
1. Generate preliminary answer from retrieved documents
2. LLM self-critique: "Is this answer complete, accurate, and well-cited?"
3. If inadequate → Retry (up to 3 times) with:
   - Different search strategy
   - Decomposed sub-queries
   - HyDE enhancement
   - Additional Knowledge Graph queries
4. After 3 failures → Trigger internet search tool
5. Generate final answer with internet sources
6. Provide feedback to vector store for continuous improvement

#### Checklist
- [ ] Design LangGraph state schema (query, tools, intermediate_steps, final_answer)
- [ ] Implement ReAct agent node (reasoning + tool selection)
- [ ] Create vector search tool with proper formatting
- [ ] Create knowledge graph query tool (SPARQL execution)
- [ ] Create HyDE tool
- [ ] Create query decomposition tool
- [ ] Create evidence synthesis tool
- [ ] Implement reflecting agent node (self-evaluation)
- [ ] Create criticism prompt (evaluate completeness, accuracy, citations)
- [ ] Implement retry logic (max 3 attempts with different strategies)
- [ ] Create internet search tool (fallback after 3 attempts)
- [ ] Test agent with 10 sample queries, verify tool usage logged

---

### 5. Advanced Search & Retrieval

**Chosen approach: MMR** — We implement MMR for its simplicity and native support in ChromaDB. Hybrid search (BM25+dense) is the alternative but requires additional infrastructure.

#### A. MMR (Maximal Marginal Relevance) — **Implemented**

| Feature | Specification | Status |
|---------|---------------|--------|
| Implementation | Query expansion + diversity reranking | [ ] |
| Lambda parameter | 0.5-0.7 (balance relevance vs. diversity) | [ ] |
| Top-k retrieval | Retrieve 50-100, rerank to top 10 | [ ] |

**Purpose:** Avoid retrieving redundant/duplicate documents while maintaining relevance

#### B. Hybrid Search — **Alternative (not implemented)**

> This approach was considered but not chosen. It requires additional infrastructure (BM25 index, RRF fusion layer) beyond what is natively supported by ChromaDB. MMR was selected instead.

| Feature | Specification | Status |
|---------|---------------|--------|
| Dense retrieval | Vector similarity (cosine distance) | N/A |
| Sparse retrieval | BM25 keyword matching — *not implemented* | N/A |
| Fusion method | RRF (Reciprocal Rank Fusion) — *not implemented* | N/A |
| Weighting | Dense 60% + Sparse 40% (tunable) | N/A |

#### C. Knowledge Graph Queries

| Feature | Specification | Status |
|---------|---------------|--------|
| Query method | SPARQL via RDFLib | [ ] |
| Tool access | Available to ReAct agent | [ ] |
| Query examples | Entity lookup, property filtering, relation traversal | [ ] |

#### Checklist
- [ ] Implement MMR reranking with configurable lambda
- [ ] Test MMR with sample query (verify diversity of results)
- [ ] Measure retrieval performance: Recall@10, Precision@10
- [ ] Create SPARQL query builder for common patterns
- [ ] Test KG queries for entity lookup, filtering, and relation traversal

---

### 6. Traceability with LangSmith

#### Configuration

| Requirement | Specification | Status |
|-----------|---------------|--------|
| Connection | LangSmith API key configured | [ ] |
| Project name | `practica2-rag` or similar | [ ] |
| Logging level | All LangGraph nodes | [ ] |

#### Metrics to Log

| Metric | Description | Status |
|--------|-------------|--------|
| Query input | Original user query | [ ] |
| Query transformations | HyDE outputs, decomposed queries | [ ] |
| Tool calls | Which tools called, with parameters | [ ] |
| Tool results | Document count, embeddings, SPARQL results | [ ] |
| Agent steps | Reasoning steps (thought/action/observation) | [ ] |
| Retry attempts | Number of retries, reasons for failure | [ ] |
| Final answer | Generated response with sources | [ ] |
| Response time | Total latency | [ ] |
| Token usage | Input + output tokens for each LLM call | [ ] |

#### Checklist
- [ ] Install `langsmith` package
- [ ] Set LangSmith API key in environment variables
- [ ] Configure LangGraph to log all node executions
- [ ] Create custom callbacks for metrics collection
- [ ] Test logging with sample query (verify all steps appear in LangSmith dashboard)
- [ ] Capture traces for at least 10 representative queries
- [ ] Document trace structure and interpretation

---

### 7. Evaluation Metrics

#### Retrieval Metrics

| Metric | Definition | Status |
|--------|-----------|--------|
| Recall@k | Fraction of relevant docs retrieved in top-k | [ ] |
| Precision@k | Fraction of top-k results that are relevant | [ ] |
| MRR (Mean Reciprocal Rank) | Average rank of first relevant result | [ ] |
| nDCG@k (Normalized DCG) | Ranking quality (position-weighted) | [ ] |

#### RAG-Specific Metrics

| Metric | Definition | Status |
|--------|-----------|--------|
| Relevance | How relevant are retrieved documents to query? | [ ] |
| Faithfulness | Is generated answer supported by retrieved docs? | [ ] |
| Answer accuracy | Does answer correctly answer the question? | [ ] |
| Citation coverage | Are sources properly cited in answer? | [ ] |

#### Evaluation Methodology

**LLM as Judge:**
- Use Claude or GPT-4 as evaluator
- Provide evaluation rubric: Relevance (1-5), Faithfulness (1-5), Accuracy (1-5)
- Aggregate scores across test set

**Test Set:**
- Create 20 gold standard Q&A pairs with expert-verified answers
- Cover diverse query types: definition, comparison, consequence, procedure
- Cover different legal topics (law texts, regulations, case law)

#### Checklist
- [ ] Create gold standard test set (20 Q&A pairs) with expert annotation
- [ ] Implement evaluation functions for each metric
- [ ] Implement LLM-as-judge prompt and scoring logic
- [ ] Run evaluation on test set with vector search baseline
- [ ] Run evaluation on test set with MMR
- [ ] Run evaluation on test set with hybrid search
- [ ] Run evaluation on test set with full ReAct+Reflecting system
- [ ] Compare baselines (Recall@5, @10; Precision@5, @10; MRR; nDCG)
- [ ] Report faithfulness and relevance scores (avg + stdev)
- [ ] Document evaluation results in technical report

---

## PART B — GENERAL SYSTEM FLOW

### Complete Query Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ USER QUERY                                                        │
│ "What are the legal consequences of ceasing payments?"           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: QUERY TRANSFORMATION                                     │
│ ├─ Detect query type (simple/complex/ambiguous)                  │
│ ├─ HyDE: Generate hypothetical documents (if ambiguous)          │
│ └─ Query Decomposition: Break into sub-queries (if complex)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: ReAct AGENT — PLANNING & REASONING                       │
│ ├─ Thought: "I need to search for legal consequences"            │
│ ├─ Action: Select tools (vector search + KG query)               │
│ ├─ Observation: Execute selected tools                           │
│ └─ Thought → Action → Observation (repeat as needed)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: RETRIEVAL (Multi-Strategy)                               │
│                                                                   │
│ Tool 1: VECTOR SEARCH (Hybrid + MMR)                             │
│ ├─ BM25 keyword search → top 50                                  │
│ ├─ Vector semantic search → top 50                               │
│ ├─ Fuse results with RRF                                         │
│ └─ Rerank with MMR → top 10 documents                            │
│                                                                   │
│ Tool 2: KNOWLEDGE GRAPH QUERIES                                  │
│ ├─ SPARQL: Find legal consequences class instances               │
│ ├─ SPARQL: Related regulations and case law                      │
│ └─ Return structured RDF triples                                 │
│                                                                   │
│ Combine document chunks + RDF triples for LLM context            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: PRELIMINARY ANSWER GENERATION                            │
│ ├─ Prompt LLM with retrieved context                             │
│ └─ Generate initial response with source citations               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: REFLECTING AGENT — SELF-EVALUATION                       │
│ ├─ Critique prompt: Check completeness, accuracy, citations      │
│ ├─ Self-assessment score (1-5)                                   │
│ └─ Decision: PASS or RETRY?                                      │
└────┬──────────────────────────────────────────┬──────────────────┘
     │                                          │
     ▼ PASS (score ≥ 4)                         ▼ FAIL (score < 4)
   ┌──────────────┐                        ┌──────────────────┐
   │ STEP 6: CALC │                        │ STEP 5B: RETRY?  │
   │   METRICS    │                        └────────┬─────────┘
   └──────┬───────┘                                 │
          │                                     [retry < 3?]
          ▼                                         │
   ┌──────────────────┐                            │
   │ Calculate Recall │                            ▼
   │ Precision, MRR   │                     ┌────────────────┐
   │ nDCG Faithfulness│                     │ Try Different: │
   │ Relevance Scores │                     │ ├─ HyDE        │
   └──────┬───────────┘                     │ ├─ Decompose   │
          │                                 │ ├─ KG queries  │
          ▼                                 │ └─ New search  │
     ┌──────────────┐                       └────────┬───────┘
     │ STEP 7: FINAL│                               │
     │ RESPONSE     │◄──────────────────────────────┘
     │ & RETURN     │ [Loop back to STEP 4]
     └──────┬───────┘
            │
            ▼
 ┌─────────────────────────┐
 │ STEP 8a: SUCCESS        │
 │ Return answer with:     │
 │ ├─ Final response       │
 │ ├─ Sources cited        │
 │ ├─ Confidence score     │
 │ ├─ Metrics              │
 │ └─ LangSmith trace link │
 └─────────────────────────┘

 OR (after 3 failed retries)

 ┌─────────────────────────┐
 │ STEP 8b: FALLBACK       │
 │ Internet Search Tool    │
 │ ├─ Search web for query │
 │ ├─ Synthesize web results
 │ ├─ Generate new answer  │
 │ └─ Provide feedback to  │
 │    knowledge base       │
 └─────────────────────────┘
```

### Detailed Step Descriptions

#### Step 1: Query Transformation
- **Input:** Raw user query
- **Processing:**
  - Detect query complexity (single-part, multi-part, ambiguous)
  - If ambiguous → apply HyDE to generate 2-3 hypothetical documents
  - If multi-part → decompose into 2-5 sub-queries
- **Output:** Processed query/queries ready for retrieval

#### Step 2: ReAct Agent Planning
- **Input:** Processed query
- **Processing:**
  - Agent thinks about retrieval strategy
  - Selects appropriate tools (vector search, KG, decomposition, HyDE)
  - Plans execution order
- **Output:** Tool selection plan

#### Step 3: Multi-Strategy Retrieval
- **Vector Search:**
  - BM25: Keyword-based retrieval (top 50)
  - Vector: Dense semantic retrieval (top 50)
  - RRF Fusion: Combine and rank
  - MMR: Rerank for diversity (top 10)

- **Knowledge Graph:**
  - SPARQL query construction based on entity/property extraction
  - Multiple SPARQL queries executed in parallel
  - Results: Structured RDF triples

- **Combination:** Merge document chunks + RDF knowledge for LLM context

#### Step 4: Preliminary Answer Generation
- **Input:** Top 10 documents + RDF triples + original query
- **Processing:**
  - Construct prompt with context + instructions for citation
  - Call LLM (Claude/GPT-4)
  - Extract answer + sources
- **Output:** Preliminary response with citations

#### Step 5: Reflecting Agent Evaluation
- **Input:** Preliminary answer + original query + retrieved documents
- **Processing:**
  - Evaluate completeness: "Does answer cover all aspects of the question?"
  - Evaluate accuracy: "Is answer supported by retrieved documents?"
  - Evaluate citation: "Are sources properly cited?"
  - Assign self-evaluation score (1-5)
- **Output:** Score + decision (PASS/RETRY)

#### Step 5B: Retry Logic (if FAIL)
- **Max retries:** 3 attempts
- **Retry strategies:**
  1. First retry: Enhanced HyDE + decomposition
  2. Second retry: Expanded KG queries + MMR reranking
  3. Third retry: Internet search tool activation

#### Step 6: Metrics Calculation
- **Input:** Query, retrieved documents, generated answer
- **Metrics:**
  - Recall@k: How many relevant docs in top-k?
  - Precision@k: How many top-k results are relevant?
  - MRR: Rank of first relevant document
  - nDCG@k: Quality of ranking
  - Faithfulness: Answer supported by documents?
  - Relevance: Documents relevant to query?

#### Step 7: Final Response
- **Components:**
  - Answer text
  - List of cited sources with page numbers
  - Confidence score (0-1)
  - Metrics (Recall@5, Precision@5, MRR, nDCG)
  - LangSmith trace link (for inspection)

#### Step 8a: Success Path
- Return structured response to user
- Log success in LangSmith
- Store feedback for continuous improvement

#### Step 8b: Fallback (Internet Search)
- Trigger after 3 failed retries
- Execute internet search for user query
- Synthesize web results with local knowledge
- Generate final answer citing both local + internet sources
- Provide feedback signal (query difficulty, coverage gaps)

---

## PART C — KNOWLEDGE GRAPH

### 1. OWL Ontology Design (Turtle Format)

#### Overview
The ontology models key concepts in Colombian business insolvency law, their relationships, and properties.

#### 1.1 Classes (Minimum 10)

| Class Name | Description | Status |
|-----------|-------------|--------|
| `EntidadJuridica` | Legal entity (top-level for parties) | [ ] |
| `Deudor` | Debtor (subClassOf EntidadJuridica) | [ ] |
| `PersonaJuridica` | Legal person/company (subClassOf Deudor) | [ ] |
| `PersonaNatural` | Natural person (subClassOf Deudor) | [ ] |
| `Acreedor` | Creditor (subClassOf EntidadJuridica) | [ ] |
| `AcreedorPrivilegiado` | Privileged creditor (subClassOf Acreedor) | [ ] |
| `AcreedorOrdinario` | Ordinary creditor (subClassOf Acreedor) | [ ] |
| `Liquidador` | Liquidator (subClassOf EntidadJuridica) | [ ] |
| `ProcedimientoInsolvencia` | Insolvency procedure (top-level) | [ ] |
| `Reorganizacion` | Reorganization (subClassOf ProcedimientoInsolvencia) | [ ] |
| `Liquidacion` | Liquidation (subClassOf ProcedimientoInsolvencia) | [ ] |
| `ActoJuridico` | Legal act | [ ] |
| `Obligacion` | Obligation | [ ] |
| `Garantia` | Guarantee | [ ] |
| `OrganoDecisorio` | Decision body | [ ] |
| `NormaLegal` | Legal norm | [ ] |

**Total Classes:** 16 (exceeds minimum of 10)

#### 1.2 Class Hierarchies (Minimum 3 rdfs:subClassOf)

```turtle
# Hierarchy 1: Procedures
ins:ProcedimientoJudicial rdfs:subClassOf ins:Proceso .

# Hierarchy 2: Reorganization vs Liquidation
ins:Reorganizacion rdfs:subClassOf ins:ProcesoInsolvencia .
ins:LiquidacionJudicial rdfs:subClassOf ins:ProcesoInsolvencia .

# Hierarchy 3: Legal Documents
ins:SentenciaJurisdiccional rdfs:subClassOf ins:DocumentoLegal .
ins:LeyInsolvencia rdfs:subClassOf ins:DocumentoLegal .
ins:DoctrinalLegal rdfs:subClassOf ins:DocumentoLegal .

# Hierarchy 4: Participants
ins:Acreedor rdfs:subClassOf ins:ParteProcesal .
ins:EmpresaInsolvente rdfs:subClassOf ins:ParteProcesal .

# Hierarchy 5: Rights
ins:DerechoCreditor rdfs:subClassOf ins:DerechoLegal .
```

**Total Hierarchies:** 5 (exceeds minimum of 3)

#### 1.3 Properties (Minimum 10)

**Object Properties (domain: URI, range: URI):**

| Property | Domain | Range | Inverse | Status |
|----------|--------|-------|---------|--------|
| `tieneAcreedor` | EmpresaInsolvente | Acreedor | acreedorDe | [ ] |
| `reguladoPor` | Reorganizacion | LeyInsolvencia | regula | [ ] |
| `regula` | LeyInsolvencia | Reorganizacion | reguladoPor | [ ] |
| `generaCesePagos` | EmpresaInsolvente | CesePagos | causadoPor | [ ] |
| `causadoPor` | CesePagos | EmpresaInsolvente | generaCesePagos | [ ] |
| `desencadenaProc` | CesePagos | ProcedimientoJudicial | desencadenaPor | [ ] |
| `fundamentadoEn` | SentenciaJurisdiccional | LeyInsolvencia | fundamenta | [ ] |
| `fundamenta` | LeyInsolvencia | SentenciaJurisdiccional | fundamentadoEn | [ ] |
| `participaEn` | ParteProcesal | ProcedimientoJudicial | tieneParte | [ ] |
| `tieneParte` | ProcedimientoJudicial | ParteProcesal | participaEn | [ ] |

**Datatype Properties (domain: URI, range: Literal):**

| Property | Domain | Range (XMLSchema) | Status |
|----------|--------|-------------------|--------|
| `nombreEmpresa` | EmpresaInsolvente | xsd:string | [ ] |
| `nitRegistro` | EmpresaInsolvente | xsd:string | [ ] |
| `fechaInsolvencia` | EmpresaInsolvente | xsd:date | [ ] |
| `montoDeuda` | Acreedor | xsd:decimal | [ ] |
| `numeroSentencia` | SentenciaJurisdiccional | xsd:string | [ ] |
| `fechaSentencia` | SentenciaJurisdiccional | xsd:dateTime | [ ] |
| `articulo` | LeyInsolvencia | xsd:int | [ ] |
| `descripcion` | Reorganizacion | xsd:string | [ ] |
| `duracionMeses` | ProcedimientoJudicial | xsd:int | [ ] |
| `costoAproximado` | ProcedimientoJudicial | xsd:decimal | [ ] |

**Total Properties:** 20 (exceeds minimum of 10)

**Property Hierarchies (Minimum 2 rdfs:subPropertyOf):**

(Note: the assignment text says rdfs:subClassOf for property hierarchies, but the correct OWL predicate for property hierarchies is rdfs:subPropertyOf)

```turtle
# Property hierarchy 1: tieneAcreedorPrivilegiado is a more specific form of tieneAcreedor
ins:tieneAcreedorPrivilegiado rdfs:subPropertyOf ins:tieneAcreedor .

# Property hierarchy 2: esSupervisadoPor is a more specific form of estaReguladoPor
ins:esSupervisadoPor rdfs:subPropertyOf ins:estaReguladoPor .
```

#### 1.4 Inverse Properties (Minimum 2 owl:inverseOf)

Already included in property table above:
- `tieneAcreedor` ↔ `acreedorDe`
- `reguladoPor` ↔ `regula`
- `generaCesePagos` ↔ `causadoPor`
- `desencadenaProc` ↔ `desencadenaPor`
- `fundamentadoEn` ↔ `fundamenta`
- `participaEn` ↔ `tieneParte`

**Total Inverse Pairs:** 6 (exceeds minimum of 2)

#### 1.5 Individuals (Minimum 4 per Class)

**PersonaJuridica** (legal company debtors):
- `ins:EmpresaA_Textil` (fictional company in Ley 1116 context)
- `ins:EmpresaB_Construccion` (fictional)
- `ins:EmpresaC_Manufacturera` (fictional)
- `ins:EmpresaD_Servicios` (fictional)
- `ins:EmpresaE_Comercio` (fictional)

**PersonaNatural** (natural person debtors):
- `ins:DeudorNatural_Juan`
- `ins:DeudorNatural_Maria`
- `ins:DeudorNatural_Carlos`
- `ins:DeudorNatural_Ana`

**Acreedor:**
- `ins:BancoNacional`
- `ins:ProveedorX`
- `ins:EmpleadosEmpresaA`
- `ins:ProveedorY`
- `ins:AcreedorPublico`

**Reorganizacion:**
- `ins:ReorganizacionEmpresaA`
- `ins:ReorganizacionEmpresaB`
- `ins:ReorganizacionEmpresaC`
- `ins:ReorganizacionExitosa`
- `ins:ReorganizacionFracaso`

**NormaLegal:**
- `ins:Ley1116_2006`
- `ins:Decreto806_2020`
- `ins:DecretoSuplemAutorizador`
- `ins:NormaReguladora_Ley1116`

**ActoJuridico:**
- `ins:Sentencia_Landmark_2015`
- `ins:Sentencia_CorteSuprema_2018`
- `ins:Sentencia_JuzgadoComercial_2019`
- `ins:Sentencia_ApelacionJurisdiccional_2020`

**Obligacion:**
- `ins:ObligacionDeuda_EmpresaA`
- `ins:ObligacionDeuda_EmpresaB`
- `ins:ObligacionDeuda_EmpresaC`
- `ins:ObligacionDeuda_Efectivo_2022`

**ProcedimientoInsolvencia:**
- `ins:Procedimiento_Ley1116_Fase1`
- `ins:Procedimiento_Ley1116_Fase2`
- `ins:Procedimiento_Liquidacion_Fase1`
- `ins:Procedimiento_Apelacion`

**Garantia:**
- `ins:GarantiaHipotecaria_EmpresaA`
- `ins:GarantiaPrendaria_EmpresaB`
- `ins:GarantiaPersonal_DeudorNatural`
- `ins:GarantiaReal_EmpresaC`

**OrganoDecisorio:**
- `ins:AsambleaAcreedores_EmpresaA`
- `ins:JuzgadoComercial_Bogota`
- `ins:SuperintendenciaDeInsolvencia`
- `ins:ComiteAcreedores_EmpresaB`

**Total Individuals:** 45+ (exceeds minimum of 4 per class × 10 classes = 40)

#### 1.6 Property Restrictions

**A. owl:allValuesFrom (Universal Quantification)**

```turtle
ins:ReorganizacionCondicional rdf:type owl:Class ;
  rdfs:subClassOf ins:Reorganizacion ;
  rdfs:subClassOf [
    rdf:type owl:Restriction ;
    owl:onProperty ins:reguladoPor ;
    owl:allValuesFrom ins:LeyInsolvencia
  ] .
# Meaning: All values for reguladoPor must be LeyInsolvencia
```

**B. owl:someValuesFrom (Existential Quantification)**

```turtle
ins:EmpresaInsolvente rdfs:subClassOf [
  rdf:type owl:Restriction ;
  owl:onProperty ins:tieneAcreedor ;
  owl:someValuesFrom ins:Acreedor
] .
# Meaning: Each EmpresaInsolvente must have at least one Acreedor
```

**C. Cardinality Restrictions**

**Option 1: owl:cardinality (exact)**
```turtle
ins:ProcedimientoUnico rdf:type owl:Class ;
  rdfs:subClassOf ins:ProcedimientoJudicial ;
  rdfs:subClassOf [
    rdf:type owl:Restriction ;
    owl:onProperty ins:tieneParte ;
    owl:cardinality 2
  ] .
# Each ProcedimientoUnico has exactly 2 parties
```

**Option 2: owl:minCardinality (minimum)**
```turtle
ins:Reorganizacion rdfs:subClassOf [
  rdf:type owl:Restriction ;
  owl:onProperty ins:participaEn ;
  owl:minCardinality 1
] .
# Each reorganization must involve at least 1 procedure
```

**Option 3: owl:maxCardinality (maximum)**
```turtle
ins:EmpresaInsolvente rdfs:subClassOf [
  rdf:type owl:Restriction ;
  owl:onProperty ins:nitRegistro ;
  owl:maxCardinality 1
] .
# Each company has at most 1 NIT (unique identifier)
```

**Total Cardinality Restrictions:** 3+ (includes min, max, exact)

#### 1.7 Logical Constructors

**A. owl:intersectionOf (AND)**

```turtle
ins:ReorganizacionExitosa rdf:type owl:Class ;
  owl:intersectionOf (
    ins:Reorganizacion
    [ rdf:type owl:Restriction ;
      owl:onProperty ins:tieneResultado ;
      owl:hasValue "EXITOSA"
    ]
  ) .
# Class combining: Reorganizacion AND ResultadoExitoso
```

**B. owl:unionOf (OR)**

```turtle
ins:ProcesoInsolvencia rdf:type owl:Class ;
  owl:unionOf (
    ins:Reorganizacion
    ins:LiquidacionJudicial
  ) .
# Any reorganization or liquidation is a proceso de insolvencia
```

**C. owl:complementOf (NOT)**

```turtle
ins:NoReorganizacion rdf:type owl:Class ;
  owl:complementOf ins:Reorganizacion .
# Everything that is not a Reorganizacion
```

**Total Logical Constructors:** 3 (intersection, union, complement)

#### 1.8 Disjoint Classes (Minimum 2 Cases)

```turtle
# Disjoint case 1: Procedures are mutually exclusive
ins:Reorganizacion owl:disjointWith ins:LiquidacionJudicial .

# Disjoint case 2: Participants
ins:Acreedor owl:disjointWith ins:EmpresaInsolvente .

# Disjoint case 3: Outcomes
ins:ReorganizacionExitosa owl:disjointWith ins:ReorganizacionFracaso .
```

**Total Disjoint Declarations:** 3 (exceeds minimum of 2)

#### 1.9 Ontology File Structure

**File:** `ontology/insolvencia_legal.ttl`

```turtle
@prefix ins: <http://www.unal.edu.co/ontologies/insolvencia#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Ontology metadata
ins: rdf:type owl:Ontology ;
  rdfs:label "Ontología de Derecho Insolvencia Colombiano"@es ;
  rdfs:comment "Modelado de conceptos en Ley 1116 de 2006 y regulaciones relacionadas"@es ;
  owl:versionInfo "1.0" ;
  rdfs:seeAlso <https://www.funcionjudicial.gob.co/> .

# Class definitions, properties, individuals follow...
```

---

### 2. Knowledge Graph Implementation

#### 2.1 GraphDB Setup

| Component | Configuration | Status |
|-----------|---------------|--------|
| GraphDB instance | Local or cloud | [ ] |
| Repository name | `insolvencia_rag` | [ ] |
| Inference | OWL2-RL or RDFS | [ ] |
| Reasoning enabled | Yes | [ ] |

#### 2.2 Ontology Upload

| Step | Action | Status |
|------|--------|--------|
| 1 | Export OWL to Turtle (.ttl) | [ ] |
| 2 | Create repository in GraphDB | [ ] |
| 3 | Upload .ttl file to GraphDB | [ ] |
| 4 | Verify triple count (expected: 500-800) | [ ] |
| 5 | Enable OWL2-RL inference | [ ] |
| 6 | Run inference & verify derived triples | [ ] |

#### 2.3 SPARQL Query Patterns

**Pattern 1: SELECT with FILTER**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?empresa ?fechaInsolvencia ?monto
WHERE {
  ?empresa rdf:type ins:EmpresaInsolvente ;
           ins:fechaInsolvencia ?fechaInsolvencia ;
           ins:montoDeuda ?monto .
  FILTER (?monto > 100000000)  # > 100M
}
ORDER BY DESC(?monto)
LIMIT 10
```

**Pattern 2: SPARQL UPDATE (INSERT)**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
INSERT DATA {
  ins:NuevaEmpresa rdf:type ins:EmpresaInsolvente ;
               ins:nombreEmpresa "Nueva Corp" ;
               ins:fechaInsolvencia "2024-03-15"^^xsd:date .
}
```

**Pattern 3: SPARQL UPDATE (DELETE)**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
DELETE WHERE {
  ?empresa rdf:type ins:EmpresaInsolvente ;
           ins:nitRegistro "123456789" .
}
```

**Pattern 4: Complex Query with OPTIONAL**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?empresa ?acreedor ?monto
WHERE {
  ?empresa rdf:type ins:EmpresaInsolvente ;
           ins:tieneAcreedor ?acreedor .
  OPTIONAL {
    ?acreedor ins:montoDeuda ?monto .
  }
}
ORDER BY DESC(?monto)
```

**Pattern 5: Aggregation with GROUP BY**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?acreedor (COUNT(?empresa) as ?numEmpresas) (SUM(?monto) as ?totalDeuda)
WHERE {
  ?empresa rdf:type ins:EmpresaInsolvente ;
           ins:tieneAcreedor ?acreedor ;
           ins:montoDeuda ?monto .
}
GROUP BY ?acreedor
ORDER BY DESC(?totalDeuda)
```

**Minimum SPARQL Operations:** 2 UPDATE + 3+ SELECT (exceeds requirement)

#### 2.4 RDFLib Integration

**Connection Method:**
```python
from rdflib import Graph, Namespace, RDF, RDFS

# Option 1: Local SPARQL endpoint
SPARQL_ENDPOINT = "http://localhost:7200/repositories/insolvencia_rag"

# Option 2: Using rdflib SPARQLStore (via HTTP)
graph = Graph('SPARQLStore', identifier="http://localhost:7200/repositories/insolvencia_rag")

# Execute SPARQL query
query = """
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?empresa WHERE {
  ?empresa rdf:type ins:EmpresaInsolvente .
}
"""
results = graph.query(query)
```

**Checklist:**
- [ ] Install GraphDB (Docker or standalone)
- [ ] Create repository `insolvencia_rag`
- [ ] Upload ontology .ttl file
- [ ] Verify ontology loaded (triple count)
- [ ] Install RDFLib Python package
- [ ] Configure SPARQL endpoint connection
- [ ] Test SELECT query (retrieve all companies)
- [ ] Test INSERT query (add new individual)
- [ ] Test UPDATE query (modify property)
- [ ] Test DELETE query (remove triple)
- [ ] Verify connection stability with 100 sequential queries
- [ ] Document 5 inference cases with GraphDB inference enabled

---

### 3. Inference Cases Documentation

#### Inference enabled: OWL2-RL in GraphDB

#### Case 1: Class Hierarchy Inference

**Rule:** `rdfs:subClassOf` transitivity
```turtle
ins:Reorganizacion rdfs:subClassOf ins:ProcesoInsolvencia .
ins:ProcesoInsolvencia rdfs:subClassOf ins:ProcesoJudicial .
# Inferred: ins:Reorganizacion rdfs:subClassOf ins:ProcesoJudicial
```

**Test:**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?clase WHERE {
  ins:Reorganizacion rdfs:subClassOf ?clase .
}
# Expected: Returns ProcesoInsolvencia, ProcesoJudicial, and owl:Thing
```

#### Case 2: Inverse Property Inference

**Rule:** `owl:inverseOf` property mirroring
```turtle
ins:tieneAcreedor owl:inverseOf ins:acreedorDe .
ins:EmpresaA_Textil ins:tieneAcreedor ins:BancoNacional .
# Inferred: ins:BancoNacional ins:acreedorDe ins:EmpresaA_Textil
```

**Test:**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?empresa WHERE {
  ins:BancoNacional ins:acreedorDe ?empresa .
}
# Expected: Returns EmpresaA_Textil (inferred from direct property)
```

#### Case 3: Property Restriction Inference

**Rule:** `owl:allValuesFrom` constraint enforcement
```turtle
ins:ReorganizacionCondicional rdfs:subClassOf [
  owl:onProperty ins:reguladoPor ;
  owl:allValuesFrom ins:LeyInsolvencia
] .
ins:ReorganizacionEmpresaA rdf:type ins:ReorganizacionCondicional ;
                       ins:reguladoPor ins:Decreto806_2020 .
# Inferred: ins:Decreto806_2020 rdf:type ins:LeyInsolvencia
```

**Test:**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?ley WHERE {
  ?ley rdf:type ins:LeyInsolvencia .
}
# Expected: Decreto806_2020 appears (inferred from allValuesFrom)
```

#### Case 4: Type Inference from Domain/Range

**Rule:** Property `domain` and `range` inference
```turtle
ins:reguladoPor rdfs:domain ins:Reorganizacion ;
             rdfs:range ins:LeyInsolvencia .
ins:ReorganizacionEmpresaA ins:reguladoPor ins:Ley1116_2006 .
# Inferred:
# - ins:ReorganizacionEmpresaA rdf:type ins:Reorganizacion
# - ins:Ley1116_2006 rdf:type ins:LeyInsolvencia
```

**Test:**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?tipo WHERE {
  ins:Ley1116_2006 rdf:type ?tipo .
}
# Expected: LeyInsolvencia type is inferred
```

#### Case 5: Logical Constructor Inference (Union)

**Rule:** `owl:unionOf` creates implicit types
```turtle
ins:ProcesoInsolvencia owl:unionOf (ins:Reorganizacion ins:LiquidacionJudicial) .
ins:ReorganizacionEmpresaA rdf:type ins:Reorganizacion .
# Inferred: ins:ReorganizacionEmpresaA rdf:type ins:ProcesoInsolvencia
```

**Test:**
```sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?proceso WHERE {
  ?proceso rdf:type ins:ProcesoInsolvencia .
}
# Expected: Returns both reorganization and liquidation instances
```

**Checklist:**
- [ ] Enable OWL2-RL reasoning in GraphDB
- [ ] Test Case 1 (class hierarchy) — verify transitive subclass inference
- [ ] Test Case 2 (inverse property) — verify bidirectional property creation
- [ ] Test Case 3 (allValuesFrom) — verify type constraint enforcement
- [ ] Test Case 4 (domain/range) — verify domain and range type inference
- [ ] Test Case 5 (unionOf) — verify implicit type assignment
- [ ] Document expected vs. actual results for each case
- [ ] Measure performance impact of inference on query time

---

### 4. Knowledge Graph RAG Architecture

#### Integration Points

```
┌──────────────────────────────────────────────────────┐
│          LLM (Claude / GPT-4)                         │
│  Reasoning, answer generation, self-evaluation        │
└────────────────┬─────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                  ▼
┌──────────────────┐  ┌─────────────────────┐
│  ReAct Agent     │  │  Reflecting Agent   │
│  (LangGraph)     │  │  Self-evaluation    │
└────────┬─────────┘  └──────────┬──────────┘
         │                       │
    Tool selection             Retry logic
    & execution                & critique
         │                       │
    ┌────┴───────────────────────┴────┐
    │                                  │
    ▼                                  ▼
┌──────────────────────────────────────────────────┐
│        Multi-Strategy Retrieval Layer             │
├──────────────────────────────────────────────────┤
│                                                   │
│ ┌────────────────────────────────────────────┐  │
│ │   Vector Store (FAISS / Chroma)            │  │
│ │  - Semantic search (embedding-based)       │  │
│ │  - MMR reranking (diversity)               │  │
│ │  - BM25 indexing (keyword search)          │  │
│ │  - Hybrid fusion (RRF)                     │  │
│ │  50K+ chunks from 50 documents             │  │
│ └────────────────────────────────────────────┘  │
│                                                   │
│ ┌────────────────────────────────────────────┐  │
│ │   Knowledge Graph Query (GraphDB)          │  │
│ │  - SPARQL execution                        │  │
│ │  - RDFLib interface                        │  │
│ │  - OWL inference enabled                   │  │
│ │  - 500-800 triples with semantics          │  │
│ └────────────────────────────────────────────┘  │
│                                                   │
│ ┌────────────────────────────────────────────┐  │
│ │   Context Fusion                           │  │
│ │  - Merge document chunks                   │  │
│ │  - Merge RDF triples                       │  │
│ │  - Organize by relevance score             │  │
│ │  - Limit context size (8K tokens)          │  │
│ └────────────────────────────────────────────┘  │
│                                                   │
└──────────────────────────────────────────────────┘
    │                        │
    ▼                        ▼
┌──────────────────────────────────────────────┐
│     Corpus (PDF Documents)                    │
│  - 50 PDFs on Colombian insolvency law       │
│  - Semantically chunked & indexed            │
│  - Indexed in vector store                   │
└──────────────────────────────────────────────┘

    │
    ▼
┌──────────────────────────────────────────────┐
│     Knowledge Base                            │
│  - Domain ontology (insolvencia_legal.ttl)   │
│  - In GraphDB with OWL2-RL inference         │
│  - SPARQL query access                       │
│  - Feedback integration                      │
└──────────────────────────────────────────────┘
```

#### Data Flow Example

**Query:** "What are the legal requirements for a company to enter reorganization under Ley 1116?"

1. **Query Transformation:**
   - Detect: Simple, single-part query
   - HyDE: Generate hypothetical document about Ley 1116 requirements

2. **ReAct Agent Planning:**
   - Thought: "I need to find legal requirements in Ley 1116"
   - Action: Call vector search + KG query tools
   - Reasoning: "Requirements are likely in law text and doctrine"

3. **Vector Search (Hybrid):**
   - BM25: Search for "requirements", "Ley 1116", "reorganization"
   - Result: Top 10 document chunks from law text
   - Semantic: Search embedding of HyDE document
   - Result: Top 10 semantically similar chunks
   - Fusion: Combine with RRF, rerank with MMR
   - Final: Top 5 diverse, relevant chunks

4. **Knowledge Graph Query:**
   - SPARQL: `SELECT ?requisito WHERE { ?req rdf:type ins:RequisitosReorganizacion . }`
   - Result: Structured requirements from ontology
   - Additional: `SELECT ?ley WHERE { ins:Reorganizacion ins:reguladoPor ?ley . }`
   - Result: Associated legal documents

5. **Context Fusion:**
   - Combine: 5 document chunks + RDF triples
   - Total context: ~3K tokens
   - Organize: Legal requirements first, then supporting doctrine

6. **Answer Generation:**
   - Prompt LLM with context + instruction for citation
   - Generate preliminary answer with source references

7. **Reflecting Agent Evaluation:**
   - Critique: "Does answer list specific legal requirements? Are sources cited?"
   - Score: 4.2/5 → PASS
   - Proceed to metrics

8. **Metrics Calculation:**
   - Recall@5: 3/3 relevant docs in top 5 = 1.0
   - Precision@5: 3/5 top results are relevant = 0.6
   - MRR: First relevant doc at rank 1 = 1.0
   - Faithfulness: All claims cited = 0.95
   - Relevance: All docs address requirements = 0.92

9. **Final Response:**
   - Answer: "Under Ley 1116, a company must meet specific requirements including: [list]"
   - Sources: "Ley 1116 de 2006, Articles X-Y; Jurisprudencia T-123/2015; Doctrine ref: [X]"
   - Confidence: 0.92
   - Metrics: Recall@5=1.0, Precision@5=0.6, MRR=1.0, nDCG@5=0.85

---

## PART D — LANGSMITH TRACEABILITY

### 1. Configuration

#### Environment Setup
```bash
export LANGSMITH_API_KEY="your_api_key_here"
export LANGSMITH_PROJECT="practica2-rag"
export LANGSMITH_TRACING_V2=true
```

#### Python Configuration
```python
import os
from langsmith import Client

# Automatic initialization
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "practica2-rag"

# Verify connection
client = Client()
print(client.list_projects())
```

**Checklist:**
- [ ] Create LangSmith account and project (`practica2-rag`)
- [ ] Generate API key
- [ ] Set environment variables
- [ ] Verify connection with test query
- [ ] Configure LangGraph to auto-log all nodes

### 2. Execution Traceability

#### Nodes to Log

| Node Name | Input | Output | Metrics |
|-----------|-------|--------|---------|
| `query_input` | User query | Processed query | Query length |
| `hyde_generation` | Query | Hypothetical docs | HyDE docs count |
| `query_decomposition` | Query | Sub-queries | Sub-query count |
| `vector_search` | Query | Document chunks | Chunk count, scores |
| `kg_query_builder` | Query | SPARQL queries | Query count |
| `kg_execution` | SPARQL | RDF triples | Triple count |
| `context_fusion` | Chunks + triples | Fused context | Context tokens |
| `answer_generation` | Context + query | Preliminary answer | Answer tokens |
| `reflecting_evaluation` | Answer + context | Score + feedback | Eval score |
| `retry_decision` | Score | Retry flag | Attempt count |
| `internet_search` | Query | Web results | Result count |
| `metrics_calculation` | Query + docs + answer | Metric scores | Recall, Precision, MRR, nDCG |
| `final_response` | All above | Structured response | Response size, confidence |

**Total Nodes:** 13 (all critical steps logged)

#### LangSmith Trace Structure

```json
{
  "trace_id": "abc123",
  "project": "practica2-rag",
  "execution_time": 4.23,
  "nodes": [
    {
      "name": "query_input",
      "start_time": 1000,
      "end_time": 1050,
      "inputs": { "query": "What are legal requirements?" },
      "outputs": { "processed_query": "..." },
      "metadata": { "query_length": 30 }
    },
    {
      "name": "vector_search",
      "start_time": 1050,
      "end_time": 1800,
      "inputs": { "query": "..." },
      "outputs": { "chunks": [...] },
      "metadata": {
        "bm25_hits": 50,
        "semantic_hits": 50,
        "fused_hits": 10,
        "avg_similarity": 0.78
      }
    },
    ...
  ],
  "execution_plan": "react_agent -> reflect_agent",
  "tools_used": ["vector_search", "kg_query", "hyde"],
  "retry_count": 0,
  "metrics": {
    "recall_at_5": 0.95,
    "precision_at_5": 0.70,
    "mrr": 1.0,
    "ndcg": 0.85
  }
}
```

**Checklist:**
- [ ] Implement LangSmith callbacks for all nodes
- [ ] Log node execution time, inputs, outputs
- [ ] Log tool usage (which tools called, parameters)
- [ ] Log intermediate steps (thoughts, actions, observations)
- [ ] Log retrieval results (chunk count, similarity scores)
- [ ] Log LLM calls (input/output tokens, model)
- [ ] Log retry attempts and reasons
- [ ] Log final metrics
- [ ] Test with 5 sample queries (verify all steps logged)
- [ ] Export traces to dashboard
- [ ] Document trace interpretation guide

### 3. Real Document Processing

#### Test Set (10+ Queries)

| # | Query | Query Type | Expected Complexity |
|---|-------|-----------|-------------------|
| 1 | "What is cessation of payments?" | Definition | Simple |
| 2 | "What are the requirements for entering reorganization?" | Procedure | Medium |
| 3 | "Compare reorganization and judicial liquidation" | Comparison | Complex |
| 4 | "What are the rights and obligations of creditors in Ley 1116?" | Multi-part | Complex |
| 5 | "How are assets distributed in judicial liquidation?" | Procedure | Medium |
| 6 | "What does Article 3 of Ley 1116 state about cessation of payments?" | Specific reference | Simple |
| 7 | "What are the main changes introduced by Decreto 806/2020?" | Legal update | Medium |
| 8 | "Explain the four-month rule for reorganization timelines" | Specific rule | Simple |
| 9 | "What jurisprudential precedents support the definition of cessation of payments?" | Jurisprudence | Complex |
| 10 | "How does the academic doctrine interpret the principle of par conditio creditorum in reorganization?" | Doctrine + theory | Complex |

**Checklist:**
- [ ] Create test query set with 10+ diverse queries
- [ ] Categorize by complexity and type
- [ ] Annotate expected answers with sources
- [ ] Run system on each query
- [ ] Capture LangSmith traces for all queries
- [ ] Export traces with metrics
- [ ] Review traces for node execution, tool usage
- [ ] Verify all nodes executed in expected order
- [ ] Check token usage for each query
- [ ] Document findings (avg latency, tool frequency, metric distribution)

---

## DELIVERABLES

### 1. Source Code
- **Location:** Root directory with subdirectories
- **Structure:**
  ```
  /
  ├── main.py (entry point, CLI)
  ├── agents/
  │   ├── react_agent.py (reasoning & tool selection)
  │   ├── reflecting_agent.py (self-evaluation & retry)
  │   └── tools.py (all tool implementations)
  ├── retrieval/
  │   ├── vector_store.py (FAISS/Chroma setup)
  │   ├── semantic_chunking.py (SemanticChunker)
  │   ├── mmr_reranking.py (MMR implementation)
  │   ├── hybrid_search.py (BM25 + vector fusion)
  │   └── kg_queries.py (SPARQL builders)
  ├── knowledge_graph/
  │   ├── graphdb_connector.py (GraphDB RDFLib setup)
  │   └── ontology_loader.py (OWL upload)
  ├── evaluation/
  │   ├── metrics.py (Recall, Precision, MRR, nDCG)
  │   ├── llm_judge.py (LLM-based evaluation)
  │   └── evaluation_runner.py (test harness)
  ├── config/
  │   ├── settings.py (API keys, paths)
  │   ├── prompts.py (system & evaluation prompts)
  │   └── schema.py (data models)
  ├── ontology/
  │   └── insolvencia_legal.ttl (OWL ontology)
  ├── content/
  │   └── docs/ (50 PDF documents)
  └── tests/
      ├── test_retrieval.py
      ├── test_agents.py
      ├── test_kg.py
      └── test_eval.py
  ```
- **Documentation:** Min 2-3 lines per function explaining purpose
- **Requirements.txt:** All dependencies pinned

#### Code Quality Checklist
- [ ] All functions have docstrings (purpose, args, returns)
- [ ] All classes documented with examples
- [ ] Error handling for all external calls (LLM, vector store, GraphDB)
- [ ] Logging at INFO/DEBUG levels
- [ ] Configuration externalized (no hardcoded paths)
- [ ] Type hints for function signatures
- [ ] Test coverage for core components (50%+ lines)
- [ ] Linting: Black formatting, flake8 checks

### 2. Document Folder
- **Location:** `content/docs/`
- **Count:** 50 PDFs minimum
- **Format:** All PDFs readable and parseable
- **Content:** Colombian insolvency law materials

#### Verification Checklist
- [ ] All 50 files present
- [ ] All files are valid PDFs (can be opened)
- [ ] All files contain extractable text (OCR if needed)
- [ ] Total document size: 50-500 MB
- [ ] File naming convention consistent (e.g., `doc_001_ley1116.pdf`)
- [ ] Create inventory spreadsheet (filename, size, pages, topic)

### 3. Technical Report (PDF)

**Filename:** `docs/TECHNICAL_REPORT.pdf`

**Sections:**

1. **System Architecture (2-3 pages)**
   - Overview diagram of all components
   - Data flow from query to answer
   - ReAct + Reflecting agent flow
   - Integration points between vector store + KG

2. **Implementation Details (3-4 pages)**
   - SemanticChunker configuration
   - Vector store (FAISS/Chroma) setup
   - MMR reranking algorithm
   - Hybrid search fusion strategy
   - SPARQL query patterns

3. **Use Cases (3+ pages, minimum 3 use cases)**

   **Use Case 1: Simple Definition Query**
   - Query: "What is cessation of payments?"
   - Expected flow: Vector search only
   - Retrieved documents: Sample chunks
   - Generated answer: Full response with citations
   - Metrics: Recall, Precision, MRR, Faithfulness
   - LangSmith trace: Nodes executed, latency

   **Use Case 2: Complex Multi-part Query**
   - Query: "What are the requirements and timelines for reorganization?"
   - Expected flow: Query decomposition + Vector search + KG queries
   - Sub-queries: Split into 2-3 parts
   - Retrieved documents: Chunks addressing each sub-query
   - Generated answer: Integrated response covering all parts
   - Metrics: Comparison of retrieval strategies
   - LangSmith trace: Tool usage, retry attempts

   **Use Case 3: Ambiguous/Short Query**
   - Query: "Cessation of payments"
   - Expected flow: HyDE + Vector search
   - HyDE documents: Generated hypothetical contexts
   - Retrieved documents: Enhanced by HyDE
   - Generated answer: Answer despite brevity
   - Metrics: Impact of HyDE on Recall/Precision
   - LangSmith trace: HyDE execution, performance improvement

4. **Evaluation Results (2-3 pages)**
   - Test set description (20 Q&A pairs)
   - Baseline results (simple vector search)
   - MMR results
   - Hybrid search results
   - Full ReAct+Reflecting results
   - Comparison table: All strategies, all metrics
   - Faithfulness and relevance scores (avg + distribution)
   - Error analysis: Cases where system underperformed

5. **Knowledge Graph Design (2 pages)**
   - Ontology overview: Classes, properties, individuals
   - Class hierarchy diagram
   - Sample instances with relationships
   - Inference rules and examples
   - SPARQL query examples with results

6. **LangSmith Traceability (1-2 pages)**
   - Trace structure and interpretation
   - Sample trace for use case query
   - Node execution timeline
   - Tool usage frequency
   - Performance bottlenecks identified
   - Metrics logged per query

7. **Conclusions and Future Work (1 page)**
   - Summary of system capabilities
   - Limitations identified
   - Recommendations for improvement
   - Potential extensions (better LLM, more documents, etc.)

#### Checklist
- [ ] Write 12+ pages technical content
- [ ] Include 5+ diagrams/screenshots
- [ ] Include 3+ use cases with end-to-end walkthrough
- [ ] Include evaluation table with 6+ metrics × 4+ strategies
- [ ] Include sample LangSmith trace screenshot
- [ ] Include ontology diagram
- [ ] Proofread for clarity and grammar
- [ ] PDF export with table of contents and page numbers

### 4. Ontology File
- **Filename:** `ontology/insolvencia_legal.ttl`
- **Format:** Turtle (RDF/OWL)
- **Content Requirements:**
  - [ ] Minimum 10 owl:Class definitions
  - [ ] Minimum 3 rdfs:subClassOf hierarchies (total 5+)
  - [ ] Minimum 10 properties with domain/range
  - [ ] Minimum 4 individuals per class (total 45+)
  - [ ] Minimum 2 owl:inverseOf cases (total 6+)
  - [ ] Property restrictions: allValuesFrom, someValuesFrom, cardinality
  - [ ] Logical constructors: intersectionOf, unionOf, complementOf
  - [ ] Minimum 2 owl:disjointWith declarations (total 3+)
  - [ ] Valid Turtle syntax, no errors

**Validation Checklist:**
- [ ] File validates as valid RDF (Turtle syntax)
- [ ] Loads successfully in GraphDB
- [ ] Triple count between 500-800
- [ ] All IRIs have proper namespace prefixes
- [ ] Comments explain complex structures
- [ ] Individuals reference existing classes

### 5. SPARQL Queries Documentation
- **Filename:** `docs/SPARQL_QUERIES.md`
- **Content:**
  - [ ] Minimum 5 documented SPARQL queries
  - [ ] Minimum 2 UPDATE queries (INSERT/DELETE)
  - [ ] Minimum 3 SELECT queries
  - [ ] Query description (purpose, expected results)
  - [ ] Query code with comments
  - [ ] Example results (sample output)
  - [ ] Use cases (when would this query be useful?)

#### Sample Query Documentation
```markdown
### Query 1: Find Insolvent Companies with Large Debts

**Purpose:** Retrieve all companies that entered insolvency with debt > 100M

**Query:**
\`\`\`sparql
PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
SELECT ?empresa ?monto
WHERE {
  ?empresa rdf:type ins:EmpresaInsolvente ;
           ins:montoDeuda ?monto .
  FILTER (?monto > 100000000)
}
ORDER BY DESC(?monto)
\`\`\`

**Expected Results:** 5-10 companies matching criteria

**Use Case:** When investigating large insolvency cases or economic impact
```

### 6. Inference Cases Documentation
- **Filename:** `docs/INFERENCE_CASES.md`
- **Content:**
  - [ ] Minimum 5 documented inference cases
  - [ ] Before/after: Explicit triples vs. inferred triples
  - [ ] Rule explained (which OWL/RDFS rule triggered)
  - [ ] Test SPARQL query showing inference
  - [ ] Example results

#### Sample Case Documentation
```markdown
## Inference Case 1: Class Hierarchy Transitive Inference

**OWL Rule:** rdfs:subClassOf transitivity

**Explicit Triples:**
```
ins:Reorganizacion rdfs:subClassOf ins:ProcesoInsolvencia .
ins:ProcesoInsolvencia rdfs:subClassOf ins:ProcesoJudicial .
```

**Inferred Triple:**
```
ins:Reorganizacion rdfs:subClassOf ins:ProcesoJudicial .
```

**Test Query:**
```sparql
SELECT ?clase WHERE {
  ins:Reorganizacion rdfs:subClassOf+ ?clase .
}
```

**Results:** ProcesoInsolvencia, ProcesoJudicial, Thing
```

### 7. 10-Minute Pitch Video
- **Platform:** YouTube (unlisted or public)
- **Duration:** 10 minutes maximum
- **Content:**
  - [ ] Problem statement (RAG for Colombian insolvency law)
  - [ ] Technical approach (ReAct + Reflecting, KG integration)
  - [ ] System demo (live query with trace)
  - [ ] Key results (metrics, evaluation)
  - [ ] Technical innovations (HyDE, KG, advanced retrieval)
  - [ ] Future work and lessons learned

**Checklist:**
- [ ] Clear audio and video quality (1080p minimum)
- [ ] Screen recording of system execution
- [ ] LangSmith dashboard shown
- [ ] Architecture diagram presented
- [ ] Sample query executed live
- [ ] Results explained with citations
- [ ] Evaluation metrics discussed
- [ ] YouTube link provided in final report

---

## EVALUATION RUBRIC

### Code (70% of grade) — 70 points

| Component | Weight | Points | Rubric |
|-----------|--------|--------|--------|
| **Advanced Retrieval (15%)** | 15 | 0-15 | MMR + Hybrid search fully implemented, tested, documented |
| **ReAct + Reflecting (15%)** | 15 | 0-15 | Agents properly orchestrated with LangGraph, retry logic, self-evaluation working |
| **Knowledge Graph (15%)** | 15 | 0-15 | OWL ontology complete (10+ classes, properties, individuals), GraphDB setup, SPARQL queries functional, 5+ inference cases |
| **System Integration (10%)** | 10 | 0-10 | Vector store + KG + agents + evaluation seamlessly integrated, no errors |
| **LangSmith Traceability (10%)** | 10 | 0-10 | All nodes logged, metrics captured, traces interpretable, sample queries traced |
| **Evaluation Metrics (5%)** | 5 | 0-5 | Recall, Precision, MRR, nDCG, Faithfulness, Relevance computed, results documented |

**Subtotal Code: 70 points**

### Documentation (20% of grade) — 20 points

| Component | Weight | Points | Rubric |
|-----------|--------|--------|--------|
| **Technical Report (12%)** | 12 | 0-12 | 12+ pages, 3+ use cases, 5+ diagrams, evaluation results, ontology design, LangSmith traces |
| **Code Comments (5%)** | 5 | 0-5 | Function docstrings, class documentation, inline comments where complex |
| **Ontology/SPARQL/Inference Docs (3%)** | 3 | 0-3 | All 3 files complete and well-documented |

**Subtotal Documentation: 20 points**

### Pitch Video (10% of grade) — 10 points

| Component | Points | Rubric |
|-----------|--------|--------|
| **Clarity & Presentation** | 3 | Clear explanation of problem and solution, good audio/video quality |
| **Technical Depth** | 4 | Demonstrates understanding of architecture, agents, KG, metrics |
| **Demo Quality** | 3 | Live system demo, LangSmith shown, results explained |

**Subtotal Pitch: 10 points**

**Total: 100 points**

---

## COMPLETION CHECKLIST

### Phase 1: Setup & Infrastructure
- [ ] Project structure created
- [ ] Virtual environment configured
- [ ] Dependencies installed (LangGraph, LangSmith, SemanticChunker, FAISS/Chroma, RDFLib, etc.)
- [ ] API keys configured (OpenAI, LangSmith, etc.)
- [ ] GraphDB instance running
- [ ] Vector store initialized
- [ ] Git repository initialized

### Phase 2: Corpus & Indexing
- [ ] All 50 PDFs verified in `content/docs/`
- [ ] Document inventory created
- [ ] SemanticChunker implemented
- [ ] Ingestion pipeline working (PDF → chunks)
- [ ] Chunks validated (semantic coherence check)
- [ ] Vector embeddings generated
- [ ] Index stored in FAISS/Chroma

### Phase 3: Retrieval Implementation
- [ ] BM25 indexing implemented
- [ ] Vector similarity search working
- [ ] RRF fusion implemented
- [ ] MMR reranking implemented
- [ ] Hybrid search wrapper created
- [ ] Retrieval metrics computed (Recall@k, Precision@k)

### Phase 4: Knowledge Graph
- [ ] Ontology designed (12 classes, 20 properties, 45+ individuals)
- [ ] Ontology exported to Turtle (.ttl)
- [ ] GraphDB repository created
- [ ] Ontology uploaded to GraphDB
- [ ] OWL2-RL inference enabled
- [ ] SPARQL queries tested (5+ queries, 2+ UPDATE)
- [ ] Inference cases documented (5+ cases)

### Phase 5: Query Processing
- [ ] HyDE implementation working
- [ ] Query decomposition detector created
- [ ] Sub-query generation implemented
- [ ] Sequential & parallel execution options working

### Phase 6: Agent Architecture
- [ ] LangGraph graph structure designed
- [ ] ReAct agent nodes implemented (Thought → Action → Observation)
- [ ] Tool selection logic working
- [ ] Vector search tool integrated
- [ ] KG query tool integrated
- [ ] HyDE tool integrated
- [ ] Query decomposition tool integrated
- [ ] Reflecting agent nodes implemented (self-evaluation, retry logic)
- [ ] Internet search tool as fallback
- [ ] Full pipeline tested end-to-end

### Phase 7: LangSmith Integration
- [ ] LangSmith project created
- [ ] API key configured
- [ ] Callbacks implemented for all nodes
- [ ] Test queries executed with tracing
- [ ] Traces verified in dashboard
- [ ] Sample traces exported

### Phase 8: Evaluation
- [ ] Test set created (20 Q&A pairs)
- [ ] Baseline evaluation (simple vector search)
- [ ] MMR evaluation
- [ ] Hybrid search evaluation
- [ ] Full system evaluation
- [ ] LLM-as-judge implementation
- [ ] Metrics computed (Recall, Precision, MRR, nDCG, Faithfulness, Relevance)
- [ ] Results documented

### Phase 9: Documentation
- [ ] Requirements.md written (this document)
- [ ] Technical report written (12+ pages)
- [ ] Code comments and docstrings added
- [ ] SPARQL queries documented
- [ ] Inference cases documented
- [ ] README.md with setup instructions

### Phase 10: Video & Final Deliverables
- [ ] Video recorded and edited (10 min)
- [ ] YouTube link obtained
- [ ] All deliverables assembled
- [ ] Final review and QA
- [ ] Submitted to course platform

---

**Document Version:** 1.0
**Last Updated:** March 28, 2026
**Author:** Ingeniería Ontológica — Universidad Nacional de Colombia
**Course:** Práctica 2 — Knowledge Graph RAG System

