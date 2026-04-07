"""
src/kg/graphdb_client.py

Cliente SPARQL ligero sobre GraphDB via SPARQLWrapper.
Provee metodos para SELECT, UPDATE y consultas de conveniencia
usadas por las herramientas del agente.

Tambien puede funcionar contra el grafo local RDFLib como fallback.
"""
from typing import List, Dict, Any

from SPARQLWrapper import SPARQLWrapper, JSON, POST

from src.config import GRAPHDB_SPARQL_ENDPOINT, GRAPHDB_UPDATE_ENDPOINT

_INS = "http://www.unal.edu.co/ontologies/insolvencia#"


class GraphDBClient:
    """Cliente SPARQL stateless para el repositorio insolvencia."""

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
        """Ejecuta un SPARQL SELECT y retorna lista de dicts con los bindings."""
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
        """Ejecuta un SPARQL UPDATE (INSERT DATA / DELETE DATA)."""
        wrapper = SPARQLWrapper(self._update_ep)
        wrapper.setMethod(POST)
        wrapper.setQuery(sparql_update)
        wrapper.query()

    # ------------------------------------------------------------------
    # Metodos de conveniencia usados por las herramientas del agente
    # ------------------------------------------------------------------
    def get_procedures_for_debtor(self, debtor_name: str) -> List[Dict]:
        """Busca procedimientos de insolvencia vinculados a un deudor por nombre."""
        safe_name = debtor_name.replace('"', '\\"')
        query = f"""
        PREFIX ins: <{_INS}>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?proc ?type ?fecha WHERE {{
            ?deudor ins:nombreRazonSocial "{safe_name}"^^xsd:string .
            ?deudor ins:iniciaEn ?proc .
            ?proc a ?type .
            OPTIONAL {{ ?proc ins:fechaAdmision ?fecha }}
        }}
        ORDER BY ?fecha
        LIMIT 10
        """
        return self.select(query)

    def get_norms_regulating_procedure(self, proc_uri: str) -> List[Dict]:
        """Busca normas que regulan un procedimiento dado."""
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
        """Retorna acreedores de un tipo especifico."""
        safe_type = creditor_type.replace('"', '').replace("'", "").replace("\\", "")
        query = f"""
        PREFIX ins: <{_INS}>
        SELECT ?acreedor ?nombre WHERE {{
            ?acreedor a ins:{safe_type} .
            OPTIONAL {{ ?acreedor ins:nombreRazonSocial ?nombre }}
        }}
        LIMIT 20
        """
        return self.select(query)

    def get_all_individuals_of_class(self, class_name: str) -> List[Dict]:
        """Retorna todos los individuos de una clase del dominio."""
        safe_class = class_name.replace('"', '').replace("'", "").replace("\\", "")
        query = f"""
        PREFIX ins: <{_INS}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?ind ?label WHERE {{
            ?ind a ins:{safe_class} .
            OPTIONAL {{ ?ind rdfs:label ?label }}
        }}
        ORDER BY ?ind
        LIMIT 50
        """
        return self.select(query)

    def get_entity_context(self, entity_local_name: str) -> List[Dict]:
        """Retorna todas las propiedades de una entidad dada por su nombre local."""
        safe_name = entity_local_name.replace('"', '').replace("'", "").replace("\\", "")
        query = f"""
        PREFIX ins: <{_INS}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?prop ?value ?valueLabel WHERE {{
            ins:{safe_name} ?prop ?value .
            OPTIONAL {{ ?value rdfs:label ?valueLabel }}
        }}
        """
        return self.select(query)
