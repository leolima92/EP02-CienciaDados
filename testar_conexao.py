from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "movielens123")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    registro = driver.execute_query("RETURN 'conectou' AS msg").records[0]
    print(registro["msg"])