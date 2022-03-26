from openie import StanfordOpenIE


class OpenIEClient:

    def __init__(self):
        properties = {
            'openie.affinity_probability_cap': 2 / 3,
        }
        self.client = StanfordOpenIE(properties=properties)

    def extract_triplets(self, text):
        return self.client.annotate(text)

# def extract_triplets(text):
#     # https://stanfordnlp.github.io/CoreNLP/openie.html#api
#     # Default value of openie.affinity_probability_cap was 1/3.
#     properties = {
#         'openie.affinity_probability_cap': 2 / 3,
#     }
#
#     with StanfordOpenIE(properties=properties) as client:
#         return client.annotate(text)
#
#
# def print_triplets(triplets):
#     for triplet in triplets:
#         print(triplet)
