from openie import StanfordOpenIE


def extract_triplets(text):
    # https://stanfordnlp.github.io/CoreNLP/openie.html#api
    # Default value of openie.affinity_probability_cap was 1/3.
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }

    with StanfordOpenIE(properties=properties) as client:
        # graph_image = 'graph.png'
        # client.generate_graphviz_graph(text, graph_image)
        return client.annotate(text)


def print_triplets(triplets):
    for triplet in triplets:
        print(triplet)
