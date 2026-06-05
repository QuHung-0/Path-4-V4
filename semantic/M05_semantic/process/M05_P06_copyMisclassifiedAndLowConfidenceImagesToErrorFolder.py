# ========= FILE NAME: M05_P06_copyMisclassifiedAndLowConfidenceImagesToErrorFolder.py =========
# FILE ROLE: Reads predictions CSV and copies wrong/low-confidence images to error analysis folders
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P06_F18 → P06_F21

# import OS utilities for path handling and directory creation
import os

# import CSV for reading prediction rows
import csv

# import shutil for copying images into analysis folders
import shutil

# import JSON for reading threshold
import json

# import shared M05 paths and logger
from semantic.M05_semantic.input.M05_I01_inputPaths import (
    M04_P01_OUTPUT_PREDICTIONS_CSV,   # predictions CSV from M04_P01
    M01_P03_OUTPUT_TEST_FOLDER,       # exact test folder from M01
    M04_P04_OUTPUT_THRESHOLD_JSON,    # confidence threshold from M04_P04

    M04_OUTPUT_FOLDER,                # M04 output root, used to derive error-analysis folder
    LOGGER,                           # logger
)

# output root for error analysis
# this folder sits alongside the M04 output folder, under a sibling directory
ERROR_ANALYSIS_ROOT = os.path.join(
    os.path.dirname(M04_OUTPUT_FOLDER), "M05_error_analysis"
)

# FUNCTION 18
def P06_F18_loadPredictionsAndThresholdFromFiles() -> tuple:
    # loads the predictions CSV and the confidence threshold JSON
    with open(M04_P01_OUTPUT_PREDICTIONS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(M04_P04_OUTPUT_THRESHOLD_JSON, "r", encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]
    LOGGER("P06", f"loaded {len(rows)} predictions, threshold={threshold}")
    return rows, threshold

# FUNCTION 19
def P06_F19_findFullImagePathInTestFolder(filename: str, test_folder: str) -> str:
    # searches all species subfolders in test folder to find the image file
    for species in os.listdir(test_folder):
        candidate = os.path.join(test_folder, species, filename)
        if os.path.exists(candidate):
            return candidate
    return None

# FUNCTION 20
def P06_F20_copyOneImageToErrorSubfolder(
    src_path: str, category: str, true_label: str, predicted_label: str
) -> None:
    # copies image into:
    # ERROR_ANALYSIS_ROOT / category / true_label__predicted_as__predicted_label /
    if category == "misclassified":
        subfolder = f"{true_label}__predicted_as__{predicted_label}"
    else:
        subfolder = f"low_conf__{true_label}"

    dst_dir = os.path.join(ERROR_ANALYSIS_ROOT, category, subfolder)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src_path, os.path.join(dst_dir, os.path.basename(src_path)))

# FUNCTION 21
def P06_F21_processAllPredictionsAndCopyErrorImages(rows: list, threshold: float, test_folder: str) -> dict:
    # iterates all predictions, copies misclassified and low-confidence images
    # now uses filename from CSV instead of guessing from folder
    stats = {"misclassified": 0, "low_confidence": 0, "not_found": 0, "total": len(rows)}

    for row in rows:
        true_label = row["true_label"]
        pred_label = row["predicted_label"]
        confidence = float(row["confidence"])
        correct = int(row["correct"])

        # get filename from CSV
        filename = row.get("filename", None)

        if not filename:
            stats["not_found"] += 1
            continue

        # find exact image path using filename
        src = P06_F19_findFullImagePathInTestFolder(filename, test_folder)

        if src is None:
            stats["not_found"] += 1
            continue

        # misclassified images go to a class-pair folder
        if not correct:
            P06_F20_copyOneImageToErrorSubfolder(src, "misclassified", true_label, pred_label)
            stats["misclassified"] += 1

        # low-confidence images are copied regardless of whether they were correct
        if confidence < threshold:
            P06_F20_copyOneImageToErrorSubfolder(src, "low_confidence", true_label, pred_label)
            stats["low_confidence"] += 1

    return stats

# MAIN
if __name__ == "__main__":
    LOGGER("P06", "=== M05_P06 — copying error images to analysis folder ===")

    LOGGER("P06", "step 1 — loading predictions and threshold...")
    rows, threshold = P06_F18_loadPredictionsAndThresholdFromFiles()

    LOGGER("P06", "step 2 — processing all predictions...")
    stats = P06_F21_processAllPredictionsAndCopyErrorImages(rows, threshold, M01_P03_OUTPUT_TEST_FOLDER)

    LOGGER("P06", "─── error analysis summary ──────────────────────")
    LOGGER("P06", f"  total predictions processed: {stats['total']}")
    LOGGER("P06", f"  misclassified images copied: {stats['misclassified']}")
    LOGGER("P06", f"  low-confidence images copied: {stats['low_confidence']}")
    LOGGER("P06", f"  images not found: {stats['not_found']}")
    LOGGER("P06", f"  output folder: {ERROR_ANALYSIS_ROOT}")
    LOGGER("P06", "─────────────────────────────────────────────────")
    LOGGER("P06", "=== M05_P06 done ===")
    LOGGER("P06", "Open the error_analysis folder and look at each subfolder:")
    LOGGER("P06", "  misclassified/X__predicted_as__Y/ = images of X that were called Y")
    LOGGER("P06", "  low_confidence/low_conf__X/       = images of X with low confidence")