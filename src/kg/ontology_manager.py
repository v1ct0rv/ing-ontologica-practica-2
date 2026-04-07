"""
src/kg/ontology_manager.py

Carga la ontologia OWL con RDFLib y expone funciones para ejecutar
consultas SPARQL SELECT sobre el grafo local.
Sirve como fallback cuando GraphDB no esta disponible.
"""
from pathlib import Path
from typing import List, Dict, Any

from rdflib import Graph, Namespace

from src.config import ONTOLOGY_PATH, ONTOLOGY_BASE_URI


INS = Namespace(ONTOLOGY_BASE_URI)

_INITNS = {
    "ins": INS,
    "owl": Namespace("http://www.w3.org/2002/07/owl#"),
    "rdf": Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    "rdfs": Namespace("http://www.w3.org/2000/01/rdf-schema#"),
    "xsd": Namespace("http://www.w3.org/2001/XMLSchema#"),
}


def load_graph(path: Path = ONTOLOGY_PATH) -> Graph:
    """Parsea el archivo Turtle de la ontologia en un grafo RDFLib."""
    g = Graph()
    g.parse(str(path), format="turtle")
    print(f"[INFO] Ontologia cargada: {len(g)} triples.")
    return g


def query_graph(graph: Graph, sparql: str) -> List[Dict[str, Any]]:
    """
    Ejecuta una consulta SPARQL SELECT sobre el grafo local RDFLib.
    Retorna una lista de dicts {nombre_variable: valor_string}.
    """
    result = graph.query(sparql, initNs=_INITNS)
    rows = []
    for row in result:
        row_dict = {}
        for var in result.vars:
            val = row[var]
            row_dict[str(var)] = str(val) if val is not None else ""
        rows.append(row_dict)
    return rows


def get_all_classes(graph: Graph) -> List[Dict[str, Any]]:
    """Retorna los labels de todas las clases OWL en la ontologia."""
    sparql = """
    SELECT ?class ?label WHERE {
        ?class a owl:Class .
        OPTIONAL { ?class rdfs:label ?label }
    }
    ORDER BY ?class
    """
    return query_graph(graph, sparql)


def get_individuals_of_class(graph: Graph, class_name: str) -> List[Dict[str, Any]]:
    """Retorna todos los individuos de una clase dada."""
    sparql = f"""
    SELECT ?ind ?label ?nombre WHERE {{
        ?ind a ins:{class_name} .
        OPTIONAL {{ ?ind rdfs:label ?label }}
        OPTIONAL {{ ?ind ins:nombreRazonSocial ?nombre }}
    }}
    ORDER BY ?ind
    """
    return query_graph(graph, sparql)


def get_entity_properties(graph: Graph, entity_uri: str) -> List[Dict[str, Any]]:
    """Retorna todas las propiedades de una entidad dada por su URI."""
    sparql = f"""
    SELECT ?prop ?value WHERE {{
        <{entity_uri}> ?prop ?value .
    }}
    """
    return query_graph(graph, sparql)
