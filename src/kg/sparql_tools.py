"""
src/kg/sparql_tools.py

Herramientas LangChain (Tools) que exponen capacidades de consulta al
Knowledge Graph para el agente ReAct. Cada tool recibe un argumento
tipo string y retorna un string formateado.

Usa el grafo local RDFLib como fuente principal, con fallback a GraphDB
si esta configurado.
"""
from langchain_core.tools import tool

from src.kg.ontology_manager import load_graph, query_graph

_graph = None  # lazy-loaded


def _get_graph():
    """Carga lazy del grafo RDFLib local."""
    global _graph
    if _graph is None:
        _graph = load_graph()
    return _graph


@tool
def query_kg_procedures(debtor_name: str) -> str:
    """
    Consulta el Knowledge Graph para encontrar procedimientos de insolvencia
    relacionados con un deudor.
    Input: nombre o razon social del deudor (ej. 'Empresa ABC S.A.S.').
    Returns: lista formateada de procedimientos con tipo y fecha de admision.
    """
    safe_name = debtor_name.replace('"', '\\"')
    sparql = f"""
    PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?proc ?procLabel ?tipo ?fecha WHERE {{
        ?deudor ins:nombreRazonSocial "{safe_name}"^^xsd:string .
        ?deudor ins:iniciaEn ?proc .
        ?proc a ?tipo .
        OPTIONAL {{ ?proc ins:fechaAdmision ?fecha }}
        OPTIONAL {{ ?proc rdfs:label ?procLabel }}
    }}
    ORDER BY ?fecha
    LIMIT 10
    """
    results = query_graph(_get_graph(), sparql)
    if not results:
        return f"No se encontraron procedimientos para el deudor: {debtor_name}"
    lines = [f"Procedimientos para '{debtor_name}':"]
    for r in results:
        label = r.get("procLabel", r.get("proc", ""))
        tipo = r.get("tipo", "").split("#")[-1]
        fecha = r.get("fecha", "sin fecha")
        lines.append(f"  - {label} (tipo: {tipo}, admision: {fecha})")
    return "\n".join(lines)


@tool
def query_kg_norms(keyword: str) -> str:
    """
    Busca normas legales (Ley/Decreto) en el Knowledge Graph por palabra clave.
    Input: palabra clave como '1116', 'decreto', 'reorganizacion'.
    Returns: lista formateada de normas encontradas.
    """
    safe_keyword = keyword.replace('"', '\\"').replace("'", "\\'").replace("\\", "\\\\")
    sparql = f"""
    PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?norma ?numero ?anio ?label WHERE {{
        ?norma a ?tipo .
        VALUES ?tipo {{ ins:Ley ins:Decreto }}
        ?norma ins:numeroNorma ?numero .
        OPTIONAL {{ ?norma ins:anoExpedicion ?anio }}
        OPTIONAL {{ ?norma rdfs:label ?label }}
        FILTER (CONTAINS(LCASE(STR(?norma)), LCASE("{safe_keyword}"))
             || CONTAINS(LCASE(?numero), LCASE("{safe_keyword}"))
             || CONTAINS(LCASE(STR(?label)), LCASE("{safe_keyword}")))
    }}
    ORDER BY DESC(?anio)
    LIMIT 10
    """
    results = query_graph(_get_graph(), sparql)
    if not results:
        return f"No se encontraron normas con: {keyword}"
    lines = [f"Normas relacionadas con '{keyword}':"]
    for r in results:
        label = r.get("label", "")
        numero = r.get("numero", "")
        anio = r.get("anio", "")
        lines.append(f"  - {label} (Num: {numero}, Año: {anio})")
    return "\n".join(lines)


@tool
def query_kg_creditors(creditor_type: str) -> str:
    """
    Recupera acreedores del Knowledge Graph por tipo.
    Input: 'AcreedorPrivilegiado' o 'AcreedorOrdinario' o 'Acreedor' para todos.
    Returns: lista formateada de nombres de acreedores.
    """
    safe_type = creditor_type.replace('"', '').replace("'", "").replace("\\", "")
    sparql = f"""
    PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
    SELECT ?acreedor ?nombre WHERE {{
        ?acreedor a ins:{safe_type} .
        OPTIONAL {{ ?acreedor ins:nombreRazonSocial ?nombre }}
    }}
    ORDER BY ?nombre
    LIMIT 20
    """
    results = query_graph(_get_graph(), sparql)
    if not results:
        return f"No se encontraron acreedores de tipo: {creditor_type}"
    lines = [f"Acreedores de tipo '{creditor_type}':"]
    for r in results:
        nombre = r.get("nombre", r.get("acreedor", "").split("#")[-1])
        lines.append(f"  - {nombre}")
    return "\n".join(lines)


@tool
def query_kg_entity(entity_name: str) -> str:
    """
    Consulta todas las propiedades de una entidad en el Knowledge Graph.
    Input: nombre local de la entidad (ej. 'Ley1116_2006', 'EmpresaABC', 'Reorganizacion').
    Returns: lista formateada de propiedades y valores.
    """
    safe_name = entity_name.replace('"', '').replace("'", "").replace("\\", "")
    sparql = f"""
    PREFIX ins: <http://www.unal.edu.co/ontologies/insolvencia#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?prop ?value ?valueLabel WHERE {{
        ins:{safe_name} ?prop ?value .
        OPTIONAL {{ ?value rdfs:label ?valueLabel }}
    }}
    """
    results = query_graph(_get_graph(), sparql)
    if not results:
        return f"No se encontro la entidad: {entity_name}"
    lines = [f"Propiedades de '{entity_name}':"]
    for r in results:
        prop = r.get("prop", "").split("#")[-1].split("/")[-1]
        value_label = r.get("valueLabel", "")
        value = value_label if value_label else r.get("value", "").split("#")[-1]
        lines.append(f"  - {prop}: {value}")
    return "\n".join(lines)


@tool
def sparql_query(sparql_query: str) -> str:
    """
    Ejecuta una consulta SPARQL SELECT arbitraria sobre la ontologia local.
    Usar solo cuando las otras herramientas KG son insuficientes.
    Input: consulta SPARQL SELECT valida.
    Returns: resultados formateados como texto.
    """
    try:
        rows = query_graph(_get_graph(), sparql_query)
        if not rows:
            return "La consulta no retorno resultados."
        lines = []
        for r in rows[:20]:
            parts = [f"{k}={v.split('#')[-1] if '#' in v else v}" for k, v in r.items()]
            lines.append(" | ".join(parts))
        return "\n".join(lines)
    except Exception as e:
        return f"Error SPARQL: {e}"


# Lista de herramientas KG para registrar en el agente
KG_TOOLS = [
    query_kg_procedures,
    query_kg_norms,
    query_kg_creditors,
    query_kg_entity,
    sparql_query,
]
