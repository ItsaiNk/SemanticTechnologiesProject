from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable


class App:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_relation(self):
        # Aura queries use an encrypted connection using the "neo4j+s" URI scheme
        with self.driver.session() as session:
            result = session.write_transaction(self._create_relation, "Harry Potter", "Ginny")
            for row in result:
                print("Created friendship between: {p1}, {p2}".format(p1=row['p1'], p2=row['p2']))

    @staticmethod
    def _create_relation(tx, subject, object):
        query = (
            "CREATE (p1:Subject { name: $subject }) "
            "CREATE (p2:Object { name: $object }) "
            "CREATE (p1)-[:LOVES]->(p2) "
            "RETURN p1, p2"
        )
        result = tx.run(query, subject=subject, object=object)
        try:
            return [{"p1": row["p1"]["name"], "p2": row["p2"]["name"]}
                    for row in result]
            # Capture any errors along with the query and data for traceability
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
