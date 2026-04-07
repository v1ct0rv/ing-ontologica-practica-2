# Guion del Video — Pitch Sustentación (10 min)
## Knowledge Graph RAG: Insolvencia Empresarial Colombiana

**Presentadores:**
- **Victor Manuel Velásquez Cabeza**
- **Jhon Alexander Ochoa Quiceno**

**Duración máxima:** 10 minutos
**Formato:** Demo en vivo + explicación técnica

---

## INTRO — Victor (0:00 - 0:45)

**[Pantalla: Slide con título del proyecto]**

> "Buenos días, profesor Guzmán. Somos Victor Velásquez y Jhon Alexander Ochoa, y vamos a presentar nuestro sistema Knowledge Graph RAG para el dominio de insolvencia empresarial colombiana."

**[Pantalla: Slide con objetivo]**

> "Construimos un sistema RAG agéntico que responde preguntas complejas en lenguaje natural sobre la Ley 1116 de 2006, el Decreto 806 de 2020 y normativa relacionada. El sistema combina búsqueda semántica, un grafo de conocimiento con ontología OWL, y un agente ReAct con reflexión, todo orquestado con LangGraph y trazado con LangSmith."

**[Pantalla: Diagrama de arquitectura del README]**

> "Esta es la arquitectura general. El flujo va de la pregunta del usuario, pasa por transformación de consulta, recuperación híbrida vector+KG, generación, reflexión, y si no es satisfactoria, reintenta hasta 3 veces antes de buscar en internet."

---

## CORPUS Y ONTOLOGÍA — Jhon (0:45 - 2:30)

**[Pantalla: Carpeta content/docs/ con los 50 PDFs]**

> "Nuestro corpus tiene 50 documentos PDF sobre insolvencia empresarial colombiana: leyes, decretos, cartillas, jurisprudencia, artículos académicos y documentos de trámite."

**[Pantalla: Terminal — python main.py --ingest (mostrar output)]**

> "El pipeline de ingesta carga los 50 PDFs con PyMuPDF, aplica chunking semántico con SemanticChunker de LangChain, y genera 4,525 chunks indexados en ChromaDB."

**[Pantalla: Protégé o editor con insolvencia.ttl]**

> "La ontología OWL está en formato Turtle y modela el dominio con 16 clases organizadas jerárquicamente. Tenemos EntidadJuridica como clase base para Deudor, Acreedor y Liquidador. ProcedimientoInsolvencia con subclases Reorganización y Liquidación. Y NormaLegal con Ley y Decreto."

**[Pantalla: Tabla de cumplimiento de requisitos del README]**

> "Cumplimos todos los requisitos de la ontología: 23 clases, 2 pares disjuntos, 18 propiedades con domain y range, 41 individuos, 2 inverseOf, restricciones allValuesFrom, someValuesFrom y cardinality, y el constructor lógico unionOf donde Deudor equivale a PersonaJuridica unión PersonaNatural."

---

## KNOWLEDGE GRAPH Y SPARQL — Victor (2:30 - 4:00)

**[Pantalla: GraphDB Workbench — repositorio insolvencia]**

> "La ontología está cargada en GraphDB con el ruleset OWL2-RL para inferencia. Veamos las consultas SPARQL."

**[Pantalla: Ejecutar en GraphDB o notebook — SELECT + ORDER BY + LIMIT]**

> "Aquí tenemos un SELECT que lista las normas legales ordenadas cronológicamente. Vemos la Ley 1116 de 2006, el CGP de 2012, los decretos de 2020 y la Ley 2445 de 2025."

**[Pantalla: Ejecutar — FILTER]**

> "Con FILTER buscamos procedimientos admitidos después de 2023. Nos retorna las dos reorganizaciones posteriores a esa fecha."

**[Pantalla: Ejecutar — INSERT DATA + DELETE DATA]**

> "Las operaciones UPDATE: insertamos un nuevo procedimiento de reorganización y vinculamos un deudor. También demostramos DELETE DATA."

**[Pantalla: Ejecutar caso de inferencia 2 — inverseOf]**

> "Y aquí lo más interesante: las inferencias. Sin inferencia, la consulta por esAcreedorDe da 0 resultados porque solo declaramos tieneAcreedor. Activando el razonador OWL2-RL, GraphDB infiere automáticamente los triples inversos y obtenemos 6 resultados."

**[Pantalla: Mostrar brevemente los 5 casos]**

> "Documentamos 5 casos de inferencia: subClassOf, dos inverseOf, subPropertyOf y equivalentClass con unionOf."

---

## TRANSFORMACIÓN DE CONSULTAS — Jhon (4:00 - 5:15)

**[Pantalla: Notebook demo — celda del Router]**

> "Cuando el usuario hace una pregunta, el sistema primero la clasifica en una de tres estrategias."

**[Ejecutar en vivo las 3 clasificaciones]**

> "Una pregunta directa como '¿Qué establece el artículo 9?' se clasifica como DIRECT. Una pregunta vaga como '¿Qué pasa cuando no se paga?' se clasifica como HYDE, donde generamos un documento hipotético. Y una pregunta compleja como '¿Requisitos de reorganización y diferencias con liquidación?' se clasifica como DECOMPOSE."

**[Pantalla: Output de HyDE]**

> "HyDE genera un fragmento de texto legal hipotético de 100-150 palabras que RESPONDERÍA la pregunta. Ese texto se usa como embedding para la búsqueda, mejorando la relevancia cuando la pregunta original es corta o ambigua."

**[Pantalla: Output de Decomposition]**

> "La descomposición divide la pregunta compleja en 2-4 sub-preguntas simples. Cada una se recupera independientemente y los resultados se fusionan."

---

## AGENTE ReAct + REFLECTING — Victor (5:15 - 7:00)

**[Pantalla: Código de graph.py — diagrama del StateGraph]**

> "El agente está implementado como un StateGraph de LangGraph con 6 nodos: query_transform, retrieve, generate, reflect y web_fallback, más el nodo de inicio."

**[Pantalla: Terminal — ejecutar consulta en vivo]**

```
python main.py "¿Qué es la cesación de pagos según la Ley 1116?"
```

> "Veamos el sistema en acción. La estrategia es DIRECT, 0 reintentos, el critic aprueba en el primer intento. La respuesta cita correctamente la Ley 1116, menciona los artículos 10 y 13, y lista 6 fuentes de documentos reales."

**[Pantalla: Terminal — ejecutar consulta compleja]**

```
python main.py "¿Cuáles son los requisitos y plazos para la reorganización y qué diferencias hay con la liquidación?"
```

> "Con una pregunta compleja, el sistema descompone en sub-preguntas, recupera por cada una, y genera una respuesta integrada."

**[Pantalla: Explicar el ciclo de reflexión]**

> "El nodo de reflexión actúa como un critic LLM que evalúa fidelidad, completitud y precisión legal. Si rechaza, el flujo vuelve a retrieve con la crítica como guía. Máximo 3 reintentos, y si aún falla, activa la búsqueda web como fallback."

---

## LANGSMITH TRAZABILIDAD — Jhon (7:00 - 8:00)

**[Pantalla: Dashboard de LangSmith — proyecto insolvencia-kg-rag]**

> "Toda la ejecución se registra automáticamente en LangSmith. Cada nodo de LangGraph aparece como un span en el trace."

**[Pantalla: Expandir un trace específico]**

> "Aquí vemos el trace completo de una consulta: el query_transform clasificó como DIRECT, el retrieve recuperó 6 chunks del vector store y consultó el Knowledge Graph, el generate sintetizó la respuesta con GPT-4o-mini, y el reflect la aprobó."

**[Pantalla: Mostrar inputs/outputs de un nodo]**

> "Podemos inspeccionar los inputs y outputs exactos de cada nodo: qué queries se transformaron, qué documentos se recuperaron, el prompt completo del LLM, y la crítica del reflector. Esto permite depurar y mejorar el sistema."

---

## MÉTRICAS Y EVALUACIÓN — Victor (8:00 - 9:15)

**[Pantalla: Terminal — python main.py --evaluate]**

> "Implementamos métricas de retrieval: Recall@k, Precision@k, MRR y nDCG. Aquí vemos los resultados sobre queries de evaluación."

**[Pantalla: Notebook — celda de LLM como juez]**

> "También implementamos evaluación con LLM como juez, que puntúa cada respuesta en tres dimensiones: relevancia, fidelidad y precisión legal, en una escala de 1 a 5."

**[Pantalla: Tabla de resultados]**

> "Las métricas confirman que el sistema recupera documentos relevantes en las primeras posiciones y genera respuestas fieles al contexto."

**[Pantalla: Terminal — pytest tests/ -v]**

> "El proyecto tiene 26 tests automatizados: 20 pasan sin API key y 6 adicionales con la API. Todos pasan satisfactoriamente."

---

## CONCLUSIONES — Jhon (9:15 - 10:00)

**[Pantalla: Slide resumen]**

> "En resumen, construimos un sistema Knowledge Graph RAG completo que:"

> "1. Procesa 50 documentos reales con chunking semántico — 4,525 chunks"
> "2. Integra una ontología OWL con 16 clases y 41 individuos en GraphDB"
> "3. Implementa un agente ReAct con reflexión y web fallback en LangGraph"
> "4. Aplica HyDE y descomposición de consultas para mejorar la recuperación"
> "5. Traza todo el razonamiento en LangSmith"
> "6. Evalúa con métricas estándar de IR y LLM como juez"

> "El sistema responde preguntas complejas sobre insolvencia empresarial colombiana con respuestas fundamentadas, citando fuentes específicas de los documentos del corpus."

**[Pantalla: Slide con nombres]**

> "Gracias. Victor Velásquez y Jhon Alexander Ochoa. ¿Alguna pregunta?"

---

## NOTAS TÉCNICAS PARA LA GRABACIÓN

### Preparación antes de grabar:
1. Tener `.env` configurado con OPENAI_API_KEY y LANGCHAIN_API_KEY
2. Tener GraphDB corriendo con el repositorio `insolvencia` cargado
3. Tener el vector store ya construido (`.chroma_db/`)
4. Abrir LangSmith en el navegador (smith.langchain.com)
5. Tener el notebook `demo.ipynb` listo con Jupyter

### Comandos a ejecutar en vivo:
```bash
# Demo ingesta (mostrar output, no ejecutar si ya existe)
python main.py --ingest

# Demo consulta directa
python main.py "¿Qué es la cesación de pagos según la Ley 1116?"

# Demo consulta compleja
python main.py "¿Cuáles son los requisitos y plazos para la reorganización y qué diferencias hay con la liquidación?"

# Demo métricas
python main.py --evaluate

# Demo tests
python -m pytest tests/ -v
```

### División de tiempos:
| Sección | Presentador | Tiempo |
|---|---|---|
| Intro + Arquitectura | Victor | 0:00 - 0:45 |
| Corpus + Ontología | Jhon | 0:45 - 2:30 |
| Knowledge Graph + SPARQL | Victor | 2:30 - 4:00 |
| Transformación de consultas | Jhon | 4:00 - 5:15 |
| Agente ReAct + Reflecting | Victor | 5:15 - 7:00 |
| LangSmith Trazabilidad | Jhon | 7:00 - 8:00 |
| Métricas y Evaluación | Victor | 8:00 - 9:15 |
| Conclusiones | Jhon | 9:15 - 10:00 |

### Distribución equitativa:
- **Victor:** Intro, KG+SPARQL, Agente, Métricas (~5 min)
- **Jhon:** Corpus+Ontología, Transformación, LangSmith, Conclusiones (~5 min)
