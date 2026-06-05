# ========= FILE NAME: M05_P03_enrichOnePredictionWithTaxonomyFromCache.py =========
# FILE ROLE: Takes a model prediction and adds WoRMS taxonomy — returns an enriched dict
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P03_F09 → P03_F11

# import JSON for loading cached taxonomy and threshold data
import json

# import M05 shared paths and logger
from semantic.M05_semantic.input.M05_I01_inputPaths import (
    M03_P03_INPUT_TAXONOMY_MASTER_JSON,  # cached taxonomy master JSON path
    M05_P03_INPUT_THRESHOLD,             # threshold JSON path from M04
    LOGGER,                              # logger
    WARN_LOGGER                          # warning logger
)

# load taxonomy cache ONCE at module import time — never re-read inside a function
_TAXONOMY_CACHE = None
_THRESHOLD = None

# helper to lazy-load taxonomy cache only when needed
def _loadTaxonomyCacheIfNeeded() -> dict:
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is None:
        with open(M03_P03_INPUT_TAXONOMY_MASTER_JSON, "r", encoding="utf-8") as f:
            _TAXONOMY_CACHE = json.load(f)
    return _TAXONOMY_CACHE

# helper to lazy-load the threshold only when needed
def _loadThresholdIfNeeded() -> float:
    global _THRESHOLD
    if _THRESHOLD is None:
        with open(M05_P03_INPUT_THRESHOLD, "r", encoding="utf-8") as f:
            data = json.load(f)
        _THRESHOLD = data["threshold"]
    return _THRESHOLD

# FUNCTION 9
def P03_F09_checkIfPredictionConfidenceMeetsThreshold(confidence: float) -> bool:
    # returns True if confidence is at or above the data-derived threshold from M04
    threshold = _loadThresholdIfNeeded()
    return confidence >= threshold

# FUNCTION 10
def P03_F10_lookUpTaxonomyForThisLabelInCachedMaster(internal_label: str) -> dict:
    # looks up the internal model label in the cached taxonomy master
    # returns the taxon dict or None if not found
    cache = _loadTaxonomyCacheIfNeeded()
    taxon = cache.get(internal_label)
    if taxon is None:
        WARN_LOGGER("P03", f"no taxonomy found for label: {internal_label}")
    return taxon

# FUNCTION 11
def P03_F11_buildEnrichedPredictionDictionaryFromResult(
    internal_label: str,
    confidence: float,
    taxon: dict,
) -> dict:
    # assembles the full enriched prediction dict — this is what gets written to JSON
    meets_threshold = P03_F09_checkIfPredictionConfidenceMeetsThreshold(confidence)

    # if prediction is low confidence or taxonomy is missing, mark it as low confidence
    if not meets_threshold or taxon is None:
        return {
            "internal_label":  internal_label,
            "confidence":      round(confidence, 6),
            "taxonomy_status": "low_confidence",
            "scientific_name": None,
            "authority":       None,
            "aphia_id":        None,
            "lineage":         None,
        }

    # otherwise return a fully verified taxonomy-enriched record
    return {
        "internal_label":  internal_label,
        "confidence":      round(confidence, 6),
        "taxonomy_status": "verified",
        "scientific_name": taxon.get("scientific_name"),
        "authority":       taxon.get("authority"),
        "aphia_id":        taxon.get("aphia_id"),
        "lineage":         taxon.get("lineage"),
    }

# MAIN
if __name__ == "__main__":
    LOGGER("P03", "=== M05_P03 — demo enrichment test ===")

    threshold = _loadThresholdIfNeeded()
    LOGGER("P03", f"confidence threshold loaded from M04: {threshold}")

    # a few example cases to show how enrichment behaves
    test_cases = [
        ("Chaetodon_lunulatus", 0.982),
        ("Dascyllus_reticulatus", 0.45),
    ]

    for label, conf in test_cases:
        taxon = P03_F10_lookUpTaxonomyForThisLabelInCachedMaster(label)
        enriched = P03_F11_buildEnrichedPredictionDictionaryFromResult(label, conf, taxon)
        LOGGER("P03", f"label={label}  conf={conf}")
        LOGGER("P03", f"  status:    {enriched['taxonomy_status']}")
        LOGGER("P03", f"  aphia:     {enriched['aphia_id']}")
        LOGGER("P03", f"  sci:       {enriched['scientific_name']}")
        LOGGER("P03", f"  authority: {enriched['authority']}")

    LOGGER("P03", "=== M05_P03 done ===")