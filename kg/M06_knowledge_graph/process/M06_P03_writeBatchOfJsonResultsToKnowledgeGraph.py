# ========= FILE NAME: M06_P03_writeBatchOfJsonResultsToKnowledgeGraph.py =========
# FILE ROLE: Reads all JSON files from M05 output and writes every record into Neo4j
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P03_F08 → P03_F10

# import OS utilities for folder traversal
import os

# import JSON for loading each inference record
import json

# import M06 shared paths, database name, and logger
from kg.M06_knowledge_graph.input.M06_I01_inputPaths import (
    M06_P03_INPUT_INFERENCE_RESULTS_FOLDER,  # folder containing M05 JSON records
    M06_NEO4J_DATABASE,                      # database name
    LOGGER,                                  # progress logger
    WARN_LOGGER                              # warning logger
)

# import connection helpers
from kg.M06_knowledge_graph.process.M06_P01_openAndCloseNeo4jDatabaseConnection import (
    P01_F01_openConnectionToNeo4jWithCredentials,        # open driver
    P01_F02_verifyConnectionIsAliveByRunningTestQuery,   # verify DB
    P01_F03_closeNeo4jConnectionAndReleaseResources,    # close driver
)

# import graph-writing helpers
from kg.M06_knowledge_graph.process.M06_P02_writeOneJsonRecordIntoKnowledgeGraph import (
    P02_F04_mergeImageNodeIntoGraph,                    # create Image node
    P02_F05_mergePredictionNodeAndLinkToImageNode,      # create Prediction node
    P02_F06_mergeSpeciesNodeAndLinkToPredictionNode,    # create Species node
    P02_F07_mergeParentTaxonChainFromSpeciesUpToKingdom,# create taxonomic chain
)

# FUNCTION 8
def P03_F08_loadAllJsonFilesFromInferenceResultsFolder(results_folder: str) -> list:
    # reads every .json file in the folder and returns a list of record dicts
    records = []
    for fname in sorted(os.listdir(results_folder)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(results_folder, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    LOGGER("P03", f"loaded {len(records)} JSON records from {results_folder}")
    return records

# FUNCTION 9
def P03_F09_writeEachJsonRecordToGraphUsingParameterisedCypher(
    records: list, driver
) -> tuple:
    # writes all records to Neo4j — each record gets its own transaction
    # uses parameterised Cypher throughout — never string formatting with user data
    written = 0
    skipped = 0
    total = len(records)

    for i, record in enumerate(records):
        try:
            with driver.session(database=M06_NEO4J_DATABASE) as session:
                P02_F04_mergeImageNodeIntoGraph(
                    session, record["image_id"], record["source_dataset"]
                )
                P02_F05_mergePredictionNodeAndLinkToImageNode(session, record)
                P02_F06_mergeSpeciesNodeAndLinkToPredictionNode(session, record)
                P02_F07_mergeParentTaxonChainFromSpeciesUpToKingdom(session, record)
            written += 1
        except Exception as e:
            WARN_LOGGER("P03", f"failed to write record {record.get('image_id','?')}: {e}")
            skipped += 1

        if (i + 1) % 50 == 0 or (i + 1) == total:
            LOGGER("P03", f"  written {i+1}/{total}  (ok={written}  skipped={skipped})")

    return written, skipped

# FUNCTION 10
def P03_F10_reportHowManyNodesAndRelationshipsWereCreated(driver) -> None:
    # queries the database and prints current node and relationship counts
    with driver.session(database=M06_NEO4J_DATABASE) as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    LOGGER("P03", f"graph now contains: {node_count} nodes, {rel_count} relationships")

# MAIN
if __name__ == "__main__":
    LOGGER("P03", "=== M06_P03 — writing all JSON records to Neo4j ===")

    # connect to Neo4j
    LOGGER("P03", "step 1 — connecting to Neo4j...")
    driver = P01_F01_openConnectionToNeo4jWithCredentials()
    alive = P01_F02_verifyConnectionIsAliveByRunningTestQuery(driver)
    if not alive:
        WARN_LOGGER("P03", "cannot reach Neo4j — is the database started in Neo4j Desktop?")
        P01_F03_closeNeo4jConnectionAndReleaseResources(driver)
        exit(1)

    # load all result JSON files produced by M05_P05
    LOGGER("P03", "step 2 — loading JSON records from M05 output...")
    records = P03_F08_loadAllJsonFilesFromInferenceResultsFolder(M06_P03_INPUT_INFERENCE_RESULTS_FOLDER)

    # write all records into the graph
    LOGGER("P03", f"step 3 — writing {len(records)} records to graph...")
    written, skipped = P03_F09_writeEachJsonRecordToGraphUsingParameterisedCypher(
        records, driver
    )

    # report current graph size
    LOGGER("P03", "step 4 — counting nodes and relationships in graph...")
    P03_F10_reportHowManyNodesAndRelationshipsWereCreated(driver)

    # close the database connection
    P01_F03_closeNeo4jConnectionAndReleaseResources(driver)

    LOGGER("P03", "─── summary ────────────────────────────────────────")
    LOGGER("P03", f"  records written:  {written}")
    LOGGER("P03", f"  records skipped:  {skipped}")
    LOGGER("P03", "────────────────────────────────────────────────────")
    LOGGER("P03", "=== M06_P03 done ===")