import os
from neo4j import GraphDatabase

_URI      = os.environ.get("NEO4J_URI",      "bolt://115.68.221.35:7687")
_USER     = os.environ.get("NEO4J_USER",     "neo4j")
_PASSWORD = os.environ.get("NEO4J_PASSWORD", "kb0614!!")
_DB       = os.environ.get("NEO4J_DB",       "neo4j")

driver = GraphDatabase.driver(_URI, auth=(_USER, _PASSWORD))


def get_session():
    return driver.session(database=_DB)
