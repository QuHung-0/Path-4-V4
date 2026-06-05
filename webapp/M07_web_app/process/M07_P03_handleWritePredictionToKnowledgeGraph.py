# ========= FILE NAME: M07_P03_handleWritePredictionToKnowledgeGraph.py =========
# FILE ROLE: Reads prediction from server session and writes it to Neo4j
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P03_F08 → P03_F10

# import UTC timestamp generator
from datetime import datetime

# import Flask redirect helpers
from flask import redirect, url_for

# import app config and logger
from webapp.M07_web_app.input.M07_I01_appConfig import (
    M07_P03_NEO4J_DATABASE,  # Neo4j database name

    LOGGER,                  # logger
    WARN_LOGGER              # warning logger
)

# import Neo4j connection helpers from M06
from kg.M06_knowledge_graph.process.M06_P01_openAndCloseNeo4jDatabaseConnection import (
    P01_F01_openConnectionToNeo4jWithCredentials,       # open driver
    P01_F03_closeNeo4jConnectionAndReleaseResources,    # close driver
)

# import graph-write helpers from M06
from kg.M06_knowledge_graph.process.M06_P02_writeOneJsonRecordIntoKnowledgeGraph import (
    P02_F04_mergeImageNodeIntoGraph,                    # create Image node
    P02_F05_mergePredictionNodeAndLinkToImageNode,      # create Prediction node
    P02_F06_mergeSpeciesNodeAndLinkToPredictionNode,    # create Species node
    P02_F07_mergeParentTaxonChainFromSpeciesUpToKingdom,# create taxonomy chain
)


# FUNCTION 8
def P03_F08_readPredictionFromServerSideSessionNotFromUserForm(session) -> dict:
    # reads the prediction that was stored server-side in P03_F07
    # this is safe — user cannot modify server session content
    result = session.get("last_prediction")
    if not result:
        WARN_LOGGER("P03", "no prediction found in session")
    return result


# FUNCTION 9
def P03_F09_enrichPredictionWithTimestampAndWriteToNeo4j(result: dict) -> bool:
    # adds a timestamp and writes the full record to Neo4j
    if not result:
        return False

    # add current UTC time to the record before writing it to the graph
    result["timestamp"] = datetime.utcnow().isoformat()

    try:
        # open Neo4j driver
        driver = P01_F01_openConnectionToNeo4jWithCredentials()

        # write one transaction in the selected Neo4j database
        with driver.session(database=M07_P03_NEO4J_DATABASE) as db_session:
            P02_F04_mergeImageNodeIntoGraph(
                db_session, result["image_id"], result["source_dataset"]
            )
            P02_F05_mergePredictionNodeAndLinkToImageNode(db_session, result)
            P02_F06_mergeSpeciesNodeAndLinkToPredictionNode(db_session, result)
            P02_F07_mergeParentTaxonChainFromSpeciesUpToKingdom(db_session, result)

        # close Neo4j driver after successful write
        P01_F03_closeNeo4jConnectionAndReleaseResources(driver)

        LOGGER("P03", f"prediction written to Neo4j — imageId: {result['image_id'][:16]}...")
        return True

    except Exception as e:
        # report Neo4j write failure
        WARN_LOGGER("P03", f"failed to write to Neo4j: {e}")
        return False


# FUNCTION 10
def P03_F10_redirectToSearchPageAfterSuccessfulWrite() -> object:
    # redirects to the search page so the user can see the record they just saved
    return redirect(url_for("search_page"))