# ========= FILE NAME: M06_P01_openAndCloseNeo4jDatabaseConnection.py =========
# FILE ROLE: Opens a connection to Neo4j, verifies it is alive, closes it cleanly
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P01_F01 → P01_F03

# import the official Neo4j driver
from neo4j import GraphDatabase

# import Neo4j settings and logger from the shared M06 input module
from kg.M06_knowledge_graph.input.M06_I01_inputPaths import (
    M06_P01_NEO4J_URI,       # connection URI
    M06_P01_NEO4J_USER,      # username
    M06_P01_NEO4J_PASSWORD,  # password
    M06_NEO4J_DATABASE,      # database name
    LOGGER,                  # normal logger
    WARN_LOGGER              # warning logger
)

# FUNCTION 1
def P01_F01_openConnectionToNeo4jWithCredentials():
    # creates a Neo4j driver using credentials from M00_C02_allSettings
    driver = GraphDatabase.driver(
        M06_P01_NEO4J_URI,
        auth=(M06_P01_NEO4J_USER, M06_P01_NEO4J_PASSWORD)
    )
    LOGGER("P01", f"driver created → {M06_P01_NEO4J_URI}")
    return driver

# FUNCTION 2
def P01_F02_verifyConnectionIsAliveByRunningTestQuery(driver) -> bool:
    # runs a simple RETURN 1 query to confirm the database is reachable
    try:
        with driver.session(database=M06_NEO4J_DATABASE) as session:
            result = session.run("RETURN 1 AS alive")
            value = result.single()["alive"]
            if value == 1:
                LOGGER("P01", "connection verified ✓ — database is alive")
                return True
    except Exception as e:
        WARN_LOGGER("P01", f"connection failed: {e}")
        return False
    return False

# FUNCTION 3
def P01_F03_closeNeo4jConnectionAndReleaseResources(driver) -> None:
    # closes the driver and releases all connection resources
    driver.close()
    LOGGER("P01", "connection closed")

# MAIN
if __name__ == "__main__":
    LOGGER("P01", "=== M06_P01 — testing Neo4j connection ===")

    # open driver
    driver = P01_F01_openConnectionToNeo4jWithCredentials()

    # verify the server responds to a test query
    alive = P01_F02_verifyConnectionIsAliveByRunningTestQuery(driver)

    if alive:
        LOGGER("P01", "Neo4j is ready — proceed to P02")
    else:
        WARN_LOGGER("P01", "Neo4j is NOT reachable — check that the database is started in Neo4j Desktop")

    # always close the connection when finished
    P01_F03_closeNeo4jConnectionAndReleaseResources(driver)
    LOGGER("P01", "=== M06_P01 done ===")