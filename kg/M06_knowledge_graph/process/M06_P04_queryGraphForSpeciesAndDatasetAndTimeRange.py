# ========= FILE NAME: M06_P04_queryGraphForSpeciesAndDatasetAndTimeRange.py =========
# FILE ROLE: Provides three query functions — by species, by dataset source, by time range
# TOTAL FUNCTIONS IN THIS FILE: 6
# FUNCTION INDEX RANGE: P04_F11 → P04_F16

# import OS utilities for output folder creation
import os

# import CSV for saving query outputs
import csv

# import M06 shared paths and logger
from kg.M06_knowledge_graph.input.M06_I01_inputPaths import (
    M06_P04_OUTPUT_QUERY_RESULTS_FOLDER,  # folder where CSV query results are stored
    M06_NEO4J_DATABASE,                  # database name
    LOGGER                               # logger
)

# import connection helpers
from kg.M06_knowledge_graph.process.M06_P01_openAndCloseNeo4jDatabaseConnection import (
    P01_F01_openConnectionToNeo4jWithCredentials,       # open connection
    P01_F02_verifyConnectionIsAliveByRunningTestQuery,  # verify DB
    P01_F03_closeNeo4jConnectionAndReleaseResources,    # close connection
)


# FUNCTION 11
def P04_F11_findAllImagesThatWerePredictedAsThisSpecies(
    driver, scientific_name: str
) -> list:
    # returns all predictions where the species scientific name matches
    # uses parameterised query — never string formatting
    with driver.session(database=M06_NEO4J_DATABASE) as session:
        result = session.run(
            """
            MATCH (img:Image)-[:HAS_PREDICTION]->(pred:Prediction)
                  -[:PREDICTS]->(s:Species {scientificName: $name})
            RETURN img.imageId        AS imageId,
                   img.sourceDataset  AS sourceDataset,
                   pred.confidence    AS confidence,
                   pred.createdAt     AS createdAt,
                   s.scientificName   AS scientificName,
                   s.authority        AS authority
            ORDER BY pred.confidence DESC
            """,
            # --- FIX: added s.authority AS authority to RETURN ---
            # OLD RETURN ended at:  s.scientificName AS scientificName
            # FIX: one line added — s.authority AS authority
            name=scientific_name,
        )
        rows = [dict(r) for r in result]
    LOGGER("P04", f"query by species '{scientific_name}': {len(rows)} results")
    return rows


# FUNCTION 12
def P04_F12_filterResultsByMinimumConfidenceScore(
    results: list, min_confidence: float
) -> list:
    # filters a result list to only rows above the minimum confidence
    filtered = [r for r in results if r.get("confidence", 0) >= min_confidence]
    LOGGER("P04", f"filtered to confidence >= {min_confidence}: {len(filtered)} results")
    return filtered


# FUNCTION 13
def P04_F13_saveQueryResultsToCSVFile(results: list, filename: str) -> None:
    # saves any list of result dicts to a CSV in the query_results folder
    os.makedirs(M06_P04_OUTPUT_QUERY_RESULTS_FOLDER, exist_ok=True)
    output_path = os.path.join(M06_P04_OUTPUT_QUERY_RESULTS_FOLDER, filename)
    if not results:
        LOGGER("P04", "no results to save")
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    LOGGER("P04", f"query results saved -> {output_path}")


# FUNCTION 14
def P04_F14_findAllPredictionsFromThisDatasetSource(
    driver, source_dataset: str
) -> list:
    # returns all predictions where the image came from a specific dataset
    with driver.session(database=M06_NEO4J_DATABASE) as session:
        result = session.run(
            """
            MATCH (img:Image {sourceDataset: $source})-[:HAS_PREDICTION]->(pred:Prediction)
            OPTIONAL MATCH (pred)-[:PREDICTS]->(s:Species)
            RETURN img.imageId         AS imageId,
                   img.sourceDataset   AS sourceDataset,
                   pred.confidence     AS confidence,
                   pred.internalLabel  AS internalLabel,
                   pred.taxonomyStatus AS taxonomyStatus,
                   pred.createdAt      AS createdAt,
                   s.scientificName    AS scientificName,
                   s.authority         AS authority
            ORDER BY pred.confidence DESC
            """,
            # --- FIX: added s.authority AS authority to RETURN ---
            # OLD RETURN: imageId, confidence, internalLabel, taxonomyStatus,
            #             scientificName  (missing sourceDataset, createdAt, authority)
            # FIX: added sourceDataset, createdAt (needed by results table)
            #      and s.authority AS authority
            source=source_dataset,
        )
        rows = [dict(r) for r in result]
    LOGGER("P04", f"query by dataset '{source_dataset}': {len(rows)} results")
    return rows


# FUNCTION 15
def P04_F15_findAllPredictionsBetweenTheseTwoDates(
    driver, from_datetime: str, to_datetime: str
) -> list:
    # returns predictions whose createdAt timestamp falls between from and to
    with driver.session(database=M06_NEO4J_DATABASE) as session:
        result = session.run(
            """
            MATCH (img:Image)-[:HAS_PREDICTION]->(pred:Prediction)
            WHERE pred.createdAt >= $from_dt AND pred.createdAt <= $to_dt
            OPTIONAL MATCH (pred)-[:PREDICTS]->(s:Species)
            RETURN img.imageId         AS imageId,
                   img.sourceDataset   AS sourceDataset,
                   pred.createdAt      AS createdAt,
                   pred.confidence     AS confidence,
                   pred.internalLabel  AS internalLabel,
                   pred.taxonomyStatus AS taxonomyStatus,
                   s.scientificName    AS scientificName,
                   s.authority         AS authority
            ORDER BY pred.createdAt DESC
            """,
            # --- FIX: added s.authority AS authority to RETURN ---
            # OLD RETURN: imageId, createdAt, confidence, internalLabel,
            #             scientificName  (missing sourceDataset, taxonomyStatus, authority)
            # FIX: added sourceDataset, taxonomyStatus (results table consistency)
            #      and s.authority AS authority
            from_dt=from_datetime,
            to_dt=to_datetime,
        )
        rows = [dict(r) for r in result]
    LOGGER("P04", f"query by time range: {len(rows)} results")
    return rows


# FUNCTION 16 — unchanged
def P04_F16_countTotalNodesAndRelationshipsInEntireGraph(driver) -> dict:
    # returns a summary dict of node and relationship counts
    with driver.session(database=M06_NEO4J_DATABASE) as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        imgs  = session.run("MATCH (n:Image) RETURN count(n) AS c").single()["c"]
        preds = session.run("MATCH (n:Prediction) RETURN count(n) AS c").single()["c"]
        specs = session.run("MATCH (n:Species) RETURN count(n) AS c").single()["c"]
    summary = {
        "total_nodes":         nodes,
        "total_relationships": rels,
        "Image_nodes":         imgs,
        "Prediction_nodes":    preds,
        "Species_nodes":       specs,
    }
    for k, v in summary.items():
        LOGGER("P04", f"  {k}: {v}")
    return summary


# MAIN
if __name__ == "__main__":
    LOGGER("P04", "=== M06_P04 — demo queries ===")

    # open a connection to Neo4j and verify it works
    driver = P01_F01_openConnectionToNeo4jWithCredentials()
    P01_F02_verifyConnectionIsAliveByRunningTestQuery(driver)

    # query 1: find all images predicted as one specific species
    LOGGER("P04", "query 1 — all predictions of Chaetodon lunulatus...")
    rows = P04_F11_findAllImagesThatWerePredictedAsThisSpecies(driver, "Chaetodon lunulatus")
    P04_F13_saveQueryResultsToCSVFile(rows, "query_chaetodon_lunulatus.csv")

    # query 2: find all predictions from one dataset source
    LOGGER("P04", "query 2 — all predictions from Fish4Knowledge dataset...")
    rows2 = P04_F14_findAllPredictionsFromThisDatasetSource(driver, "Fish4Knowledge")
    LOGGER("P04", f"  total Fish4Knowledge predictions in graph: {len(rows2)}")

    # query 3: show graph summary
    LOGGER("P04", "query 3 — graph summary...")
    P04_F16_countTotalNodesAndRelationshipsInEntireGraph(driver)

    # close connection cleanly
    P01_F03_closeNeo4jConnectionAndReleaseResources(driver)
    LOGGER("P04", "=== M06_P04 done ===")