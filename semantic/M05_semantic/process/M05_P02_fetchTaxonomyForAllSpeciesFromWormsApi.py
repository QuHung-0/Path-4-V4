# ========= FILE NAME: M05_P02_fetchTaxonomyForAllSpeciesFromWormsApi.py =========
# FILE ROLE: Calls WoRMS REST API once per species, saves full taxonomy tree to a master JSON
# TOTAL FUNCTIONS IN THIS FILE: 5
# FUNCTION INDEX RANGE: P02_F04 → P02_F08

# import OS utilities for saving output files
import os

# import JSON for loading and saving taxonomy data
import json

# import time so we can sleep between API calls
import time

# import requests for HTTP calls to the WoRMS REST API
import requests

# import the shared M05 inputs, outputs, API settings, and logger
from semantic.M05_semantic.input.M05_I01_inputPaths import (
    M05_P02_INPUT_LABEL_SCIENTIFIC_JSON,   # input label→scientific map from M05_P01
    M05_P02_OUTPUT_TAXONOMY_MASTER_JSON,   # output cache with full taxonomy

    M05_P02_WORMS_BASE_URL,               # WoRMS REST base URL
    M05_P02_WORMS_API_SLEEP_SECONDS,      # delay between requests

    LOGGER,                               # normal logger
    WARN_LOGGER                           # warning logger
)

# FUNCTION 4
def P02_F04_askWormsApiForThisSpeciesAcceptedRecord(scientific_name: str) -> dict:
    # calls WoRMS AphiaRecordsByName endpoint and returns the first accepted record found
    url = f"{M05_P02_WORMS_BASE_URL}/AphiaRecordsByName/{requests.utils.quote(scientific_name)}"
    params = {"like": "false", "marine_only": "true"}  # exact match only, marine species only
    resp = requests.get(url, params=params, timeout=15) # send request with a timeout

    # if the request failed or returned no JSON data, report failure
    if resp.status_code != 200 or not resp.json():
        WARN_LOGGER("P02", f"no record found for: {scientific_name}")
        return {}

    records = resp.json()  # WoRMS may return a list of possible matches

    # prefer accepted records, skip synonyms
    for rec in records:
        if rec.get("status") == "accepted":
            return rec

    # fallback: follow valid_AphiaID if first record is a synonym
    first = records[0]
    valid_id = first.get("valid_AphiaID")
    if valid_id:
        fallback_url = f"{M05_P02_WORMS_BASE_URL}/AphiaRecordByAphiaID/{valid_id}"
        fallback_resp = requests.get(fallback_url, timeout=15)
        if fallback_resp.status_code == 200:
            return fallback_resp.json()

    # if no better record exists, return the first record
    return first

# FUNCTION 5
def P02_F05_extractAphiaIdAndLineageFromWormsRecord(record: dict) -> dict:
    # pulls AphiaID, authority, and the full 7-level Linnaean lineage from a WoRMS record
    if not record:
        return {}

    lineage = []
    # these field names match exactly what the WoRMS API returns in the JSON response
    rank_fields = ["kingdom", "phylum", "class", "order", "family", "genus", "scientificname"]
    rank_names  = ["Kingdom", "Phylum",  "Class", "Order", "Family", "Genus", "Species"]

    for field, rank in zip(rank_fields, rank_names):
        value = record.get(field)
        if value:
            lineage.append({"rank": rank, "scientificname": value})

    return {
        "aphia_id":       record.get("AphiaID"),
        "scientific_name": record.get("scientificname"),
        "authority":      record.get("authority"),
        "status":         record.get("status"),
        "lineage":        lineage,
    }

# FUNCTION 6
def P02_F06_waitOneSecondBetweenApiCallsToRespectRateLimit() -> None:
    # WoRMS allows roughly 10 requests/minute, so sleeping 1 second keeps us safe
    time.sleep(M05_P02_WORMS_API_SLEEP_SECONDS)

# FUNCTION 7
def P02_F07_fetchAndCacheTaxonomyForAllSpecies(
    label_scientific_json: str, output_json: str
) -> dict:
    # loads label→scientific map, fetches WoRMS taxonomy for each species, saves master JSON
    with open(label_scientific_json, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    taxonomy_master = {}  # final cache for all species taxonomy records

    # iterate through each internal label and its scientific name
    for internal_label, scientific_name in sorted(lookup.items()):
        WARN_LOGGER("P02", f"fetching: {scientific_name}...")

        # query WoRMS for the species record
        record = P02_F04_askWormsApiForThisSpeciesAcceptedRecord(scientific_name)

        # extract only the useful parts of the response
        taxon = P02_F05_extractAphiaIdAndLineageFromWormsRecord(record)

        if taxon:
            taxonomy_master[internal_label] = taxon
            WARN_LOGGER("P02", f"  AphiaID: {taxon['aphia_id']}  status: {taxon['status']}")
            WARN_LOGGER("P02", f"  lineage: {' → '.join(t['scientificname'] for t in taxon['lineage'])}")
        else:
            WARN_LOGGER("P02", f"  failed to fetch taxonomy for {scientific_name}")
            taxonomy_master[internal_label] = None

        # sleep between requests so the API is not hammered
        P02_F06_waitOneSecondBetweenApiCallsToRespectRateLimit()

    # save the complete taxonomy cache to disk
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(taxonomy_master, f, indent=4, sort_keys=True)

    WARN_LOGGER("P02", f"taxonomy master saved → {output_json}")
    return taxonomy_master

# FUNCTION 8
def P02_F08_loadCachedTaxonomyMasterFromJsonFile(taxonomy_json_path: str) -> dict:
    # loads the cached taxonomy master so other modules can use it without API calls
    with open(taxonomy_json_path, "r", encoding="utf-8") as f:
        master = json.load(f)
    LOGGER("P02", f"taxonomy master loaded — {len(master)} entries")
    return master

# MAIN
if __name__ == "__main__":
    LOGGER("P02", "=== M05_P02 — fetching taxonomy from WoRMS API ===")
    LOGGER("P02", "NOTE: this calls the internet — needs network connection")
    LOGGER("P02", "NOTE: runs once then caches — never needs to run again")

    LOGGER("P02", "step 1 — fetching and caching all 10 species...")
    taxonomy_master = P02_F07_fetchAndCacheTaxonomyForAllSpecies(
        M05_P02_INPUT_LABEL_SCIENTIFIC_JSON, M05_P02_OUTPUT_TAXONOMY_MASTER_JSON
    )

    LOGGER("P02", f"step 2 — verifying cache...")
    found = sum(1 for v in taxonomy_master.values() if v is not None)
    missing = sum(1 for v in taxonomy_master.values() if v is None)
    LOGGER("P02", f"  successfully fetched: {found}/10")
    if missing:
        WARN_LOGGER("P02", f"  failed to fetch: {missing} species — check network and re-run")
    else:
        LOGGER("P02", "  all 10 species fetched successfully ✓")

    LOGGER("P02", "=== M05_P02 done ===")