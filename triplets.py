from openie import StanfordOpenIE


class OpenIEClient:

    def __init__(self):
        properties = {
            'openie.affinity_probability_cap': 0.8,
        }
        self.client = StanfordOpenIE(properties=properties)

    def extract_triplets(self, text):
        return self.client.annotate(text)
