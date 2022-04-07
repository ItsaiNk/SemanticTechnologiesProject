from rdflib import Graph, URIRef
from SPARQL_query import query
from config import uri, database_name, auth_pwd, auth_user


def create_graph():
    g = Graph(store="Neo4j")
    config = {'uri': uri, 'database': database_name,
              'auth': {'user': auth_user, 'pwd': auth_pwd}}
    g.open(config, create=False)
    return g


def _create_uri(string, dict_elements={}):
    base_uri = "http://example.com/HP#"
    el = str(string).lower().title()
    if el in dict_elements.keys():
        el = dict_elements[el]
    res = query(el)
    if res is not None:
        return URIRef(str(res))
    else:
        return URIRef(base_uri + str(string).lower().title().replace(" ", "_"))


def add_triple(graph, s, p, o, dict_elements={}):
    graph.add((_create_uri(s, dict_elements), _create_uri(p, dict_elements), _create_uri(o, dict_elements)))
