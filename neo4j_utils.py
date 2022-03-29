import rdflib
from rdflib import Graph


def create_graph():
    g = Graph(store="Neo4j")
    theconfig = {'uri': "neo4j://localhost:7687", 'database': 'harrypotter', 'auth': {'user': "neo4j", 'pwd': "4218465"}}
    g.open(theconfig, create=False)
    return g


def add_triple(graph, s, p, o):
    base_uri = "http://example.com/HP#"
    s = rdflib.URIRef(base_uri + str(s).replace(" ", "_"))
    p = rdflib.URIRef(base_uri + str(p).replace(" ", "_"))
    o = rdflib.URIRef(base_uri + str(o).replace(" ", "_"))
    graph.add((s, p, o))


def print_graph(graph):
    for s, p, o in graph:
        print(s, p, o)
