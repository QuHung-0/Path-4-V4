# ========= FILE NAME: M05_P04_buildAndWriteJsonRecordForOneInference.py =========
# FILE ROLE: Wraps one enriched prediction into a complete JSON record and saves it to disk
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P04_F12 → P04_F14

# import OS utilities for file paths
import os

# import hashing utility for generating unique image IDs
import hashlib

# import JSON for writing output records
import json

# import UTC datetime so each record is time-stamped
from datetime import datetime

# FUNCTION 12
def P04_F12_generateContentHashFromImageFileBytes(image_path: str) -> str:
    # reads the image file as raw bytes and computes SHA-256 hash
    # same image file always gives same hash — enables true deduplication in Neo4j
    with open(image_path, "rb") as f:
        file_bytes = f.read()
    return hashlib.sha256(file_bytes).hexdigest()

# FUNCTION 13
def P04_F13_assembleCompleteJsonRecordFromAllFields(
    image_id: str,
    image_path: str,
    enriched: dict,
) -> dict:
    # combines image identity, prediction, and taxonomy into one complete record
    return {
        "image_id":        image_id,                            # unique content hash for the image
        "image_filename":  os.path.basename(image_path),        # file name only, without folder path
        "internal_label":  enriched["internal_label"],          # model label
        "confidence":      enriched["confidence"],              # model confidence
        "taxonomy_status": enriched["taxonomy_status"],         # verified or low_confidence
        "scientific_name": enriched["scientific_name"],         # scientific name if verified
        "authority":       enriched.get("authority"),           # AphiaID author
        "aphia_id":        enriched["aphia_id"],                # WoRMS AphiaID if verified
        "lineage":         enriched["lineage"],                 # taxonomy lineage if verified
        "timestamp":       datetime.utcnow().isoformat(),       # record creation time in UTC
        "model_version":   "best_model.pth",                    # model file used for inference
        "source_dataset":  "Fish4Knowledge",                    # source dataset label
    }

# FUNCTION 14
def P04_F14_writeJsonRecordToResultsFolder(record: dict, output_result_folder: str) -> str:
    # saves the record as a JSON file named by the image_id (SHA-256 hash)
    os.makedirs(output_result_folder, exist_ok=True)  # create folder if needed
    output_path = os.path.join(output_result_folder, f"{record['image_id']}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4)
    return output_path