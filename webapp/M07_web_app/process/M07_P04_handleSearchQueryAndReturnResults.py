# ========= FILE NAME: M07_P04_handleSearchQueryAndReturnResults.py =========
# FILE ROLE: Reads search filters from URL and queries Neo4j using parameterised Cypher
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P04_F11 → P04_F13

# import Flask-independent shared config and logger
from webapp.M07_web_app.input.M07_I01_appConfig import (
    M07_P03_NEO4J_DATABASE,  # Neo4j database name

    LOGGER,                  # logger
)

# import Neo4j connection helpers
from kg.M06_knowledge_graph.process.M06_P01_openAndCloseNeo4jDatabaseConnection import (
    P01_F01_openConnectionToNeo4jWithCredentials,       # open driver
    P01_F03_closeNeo4jConnectionAndReleaseResources,    # close driver
)

# import query helpers from M06
from kg.M06_knowledge_graph.process.M06_P04_queryGraphForSpeciesAndDatasetAndTimeRange import (
    P04_F11_findAllImagesThatWerePredictedAsThisSpecies,  # query by species
    P04_F14_findAllPredictionsFromThisDatasetSource,      # query by dataset source
    P04_F15_findAllPredictionsBetweenTheseTwoDates,       # FIX 5: import time-range query
)


# FUNCTION 11
def P04_F11_readSearchFiltersFromQueryParameters(request) -> dict:
    # reads optional filters from URL query string — all are optional
    return {
        "species":    request.args.get("species",    "").strip(),  # species name filter
        "dataset":    request.args.get("dataset",    "").strip(),  # dataset source filter
        "min_conf":   request.args.get("min_conf",   "").strip(),  # minimum confidence filter
        "from_date":  request.args.get("from_date",  "").strip(),  # FIX 4: start of date range
        "to_date":    request.args.get("to_date",    "").strip(),  # FIX 4: end of date range
    }


# FUNCTION 12
def P04_F12_callParameterisedCypherQueryWithUserFilters(filters: dict) -> list:
    # selects which query to run based on which filters were provided

    # open Neo4j connection
    driver  = P01_F01_openConnectionToNeo4jWithCredentials()
    results = []

    # read filters out of the dict
    species   = filters.get("species")
    dataset   = filters.get("dataset")
    min_conf  = filters.get("min_conf")
    from_date = filters.get("from_date")   # FIX 5: read date filters
    to_date   = filters.get("to_date")     # FIX 5: read date filters

    try:
        # FIX 5: time-range branch must come BEFORE species branch so it is not shadowed
        if from_date or to_date:
            # convert bare dates to ISO datetime strings Neo4j can compare
            from_dt = f"{from_date}T00:00:00" if from_date else "2000-01-01T00:00:00"
            to_dt   = f"{to_date}T23:59:59"   if to_date   else "2099-12-31T23:59:59"
            results = P04_F15_findAllPredictionsBetweenTheseTwoDates(driver, from_dt, to_dt)

        # if species filter exists, query by species
        elif species:
            results = P04_F11_findAllImagesThatWerePredictedAsThisSpecies(driver, species)

        # otherwise if dataset filter exists, query by source
        elif dataset:
            results = P04_F14_findAllPredictionsFromThisDatasetSource(driver, dataset)

        # otherwise return a default query with no filter and a cap of 100 rows
        else:
            with driver.session(database=M07_P03_NEO4J_DATABASE) as db_session:
                raw = db_session.run(
                    """
                    MATCH (img:Image)-[:HAS_PREDICTION]->(pred:Prediction)
                    OPTIONAL MATCH (pred)-[:PREDICTS]->(s:Species)
                    RETURN img.imageId        AS imageId,
                           img.sourceDataset  AS sourceDataset,
                           pred.confidence    AS confidence,
                           pred.internalLabel AS internalLabel,
                           pred.taxonomyStatus AS taxonomyStatus,
                           pred.createdAt     AS createdAt,
                           s.scientificName   AS scientificName,
                           s.authority         AS authority
                    ORDER BY pred.createdAt DESC
                    LIMIT 100
                    """
                )
                results = [dict(r) for r in raw]

        # apply confidence filter if provided
        if min_conf:
            try:
                cutoff  = float(min_conf)
                results = [
                    r for r in results
                    if r.get("confidence") is not None and r["confidence"] >= cutoff
                ]
            except ValueError:
                # ignore invalid numeric input rather than crashing
                pass

    finally:
        # always close the database connection
        P01_F03_closeNeo4jConnectionAndReleaseResources(driver)

    LOGGER("P04", f"search returned {len(results)} results")
    return results


# FUNCTION 13
def P04_F13_renderSearchResultsPageWithQueryResults(results: list, filters: dict) -> dict:
    # packages results and filters into a dict for the template to render
    return {
        "results":      results,           # list of result rows
        "filters":      filters,           # active filters (now includes from_date / to_date)
        "result_count": len(results),      # number of matches
    }