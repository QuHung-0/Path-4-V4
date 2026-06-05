# ========= FILE NAME: M07_P06_runBulkPredictOnRemainingImages.py =========
# FILE ROLE: Loads the trained ResNet-18 and runs inference on every image
#            in the M01_P05 output folder (both known_species and excluded_species).
#
#            For known_species images:
#              - true label = species folder name
#              - is_correct = (predicted_label == true_label)
#              - this tests model generalisation on unseen images of trained species
#
#            For excluded_species images:
#              - true label = fish_XX (model has never seen this species)
#              - is_correct = N/A — model will force a prediction into 10 known classes
#              - this shows which known species the model confuses unknown fish with
#
# TOTAL FUNCTIONS IN THIS FILE: 5
# FUNCTION INDEX RANGE: P06_F18 → P06_F22
# RUN ORDER: after M01_P05

import os
import csv
import json
from collections import Counter

import torch

# ── output path constants ──────────────────────────────────────────────────────
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M01_P05_OUTPUT_REMAINING_FOLDER,     # root of M01_P05 output — known_species/ and excluded_species/
    M07_P06_OUTPUT_BULK_PREDICT_FOLDER,  # output folder for this module's CSV
    M07_P06_OUTPUT_BULK_PREDICTIONS_CSV, # full path for the output CSV
    M05_P05_INPUT_BEST_MODEL_PTH,        # trained best model weights (same weights used in M05_P05)
    M05_P05_INPUT_CLASS_TO_IDX_JSON,     # class name <-> numeric index mapping
    M05_P05_INPUT_NORM_STATS_JSON,       # normalization stats from M02
)

# ── image size setting ─────────────────────────────────────────────────────────
from config.M00_pipeline_config.M00_C02_allSettings import (
    M02_P01_P02_INPUT_IMAGE_SIZE as M07_P06_INPUT_IMAGE_SIZE,  # 224 — same size used everywhere
)

# ── logger ─────────────────────────────────────────────────────────────────────
from config.M00_pipeline_config.M00_C03_logger import (
    LG00_F01_printStepToConsole    as LOGGER,
    LG00_F02_printWarningToConsole as WARN_LOGGER,
)

# ── transform helpers (M02_P02) ────────────────────────────────────────────────
from dataset.M02_preprocessing.process.M02_P02_defineAllTransformPipelines import (
    P02_F04_loadNormalizationStatsFromJsonFile,    # loads mean/std from JSON file
    P02_F07_buildInferenceTransformForSingleImage, # builds inference-only transform (no augmentation)
)

# ── class-map loader (M03_P03) ────────────────────────────────────────────────
from model_weights.M03_model.process.M03_P03_buildDataloadersForTrainingAndValidation import (
    P03_F11_loadClassToIndexMappingFromJsonFile,   # loads {class_name: idx} and returns {idx: class_name}
)

# ── model loader + inference helpers (M03_P07) ────────────────────────────────
from model_weights.M03_model.process.M03_P07_loadModelAndPredictOneSingleImage import (
    P07_F23_loadTrainedModelWeightsFromFile,                # builds ResNet-18 then loads saved weights
    P07_F24_preprocessOneSingleImageForInference,           # opens image file -> normalised [1,3,224,224] tensor
    P07_F25_runForwardPassAndGetSoftmaxProbabilities,       # forward pass -> softmax probability vector
    P07_F26_convertSoftmaxOutputToLabelAndConfidenceScore,  # probs -> (label_string, confidence_float)
)

# ── taxonomy enrichment (M05_P03) ─────────────────────────────────────────────
from semantic.M05_semantic.process.M05_P03_enrichOnePredictionWithTaxonomyFromCache import (
    P03_F10_lookUpTaxonomyForThisLabelInCachedMaster,       # cache lookup: label -> taxon dict or None
    P03_F11_buildEnrichedPredictionDictionaryFromResult,    # builds verified/low_confidence enriched dict
)

# accepted image extensions — same set as M01_P01
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


# FUNCTION 18
def P06_F18_loadModelAndTransformForBulkInference() -> tuple:
    # loads model, transform, and idx_to_class mapping once before the inference loop
    # returns (model, transform, idx_to_class, device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER("P06", f"device: {device}")

    # load normalization stats and build inference-only transform
    # this is the exact same pattern used in M05_P05 __main__
    mean, std = P02_F04_loadNormalizationStatsFromJsonFile(M05_P05_INPUT_NORM_STATS_JSON)
    transform = P02_F07_buildInferenceTransformForSingleImage(mean, std, M07_P06_INPUT_IMAGE_SIZE)

    # load class index -> label mapping
    idx_to_class = P03_F11_loadClassToIndexMappingFromJsonFile(M05_P05_INPUT_CLASS_TO_IDX_JSON)

    # P07_F23 builds ResNet-18, replaces the final layer, loads weights, and calls model.eval()
    # all in one call — no need to call P02_F04 or P02_F05 separately here
    model = P07_F23_loadTrainedModelWeightsFromFile(M05_P05_INPUT_BEST_MODEL_PTH, device)

    LOGGER("P06", f"model ready — {len(idx_to_class)} classes")
    return model, transform, idx_to_class, device


# FUNCTION 19
def P06_F19_collectAllImagePathsFromRemainingFolder() -> list:
    # walks the M01_P05 output folder and collects every image path
    # returns a list of dicts: {image_path, category, folder_name}
    # category = "known_species" or "excluded_species"
    # folder_name = species_name for known, fish_XX for excluded

    records = []

    for category in ("known_species", "excluded_species"):
        category_dir = os.path.join(M01_P05_OUTPUT_REMAINING_FOLDER, category)
        if not os.path.isdir(category_dir):
            WARN_LOGGER("P06", f"folder not found, skipping: {category_dir}")
            continue

        for folder_name in sorted(os.listdir(category_dir)):
            folder_path = os.path.join(category_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for fname in sorted(os.listdir(folder_path)):
                if os.path.splitext(fname)[1].lower() not in _IMAGE_EXTS:
                    continue
                records.append({
                    "image_path":  os.path.join(folder_path, fname),
                    "category":    category,
                    "folder_name": folder_name,
                })

    LOGGER("P06", f"found {len(records)} images to predict")
    return records


# FUNCTION 20
def P06_F20_predictOneImageAndReturnResultRow(
    image_path:   str,
    category:     str,
    folder_name:  str,
    model,
    transform,
    idx_to_class: dict,
    device,
) -> dict:
    # runs the full predict -> enrich pipeline on one image and returns one CSV row dict

    try:
        tensor       = P07_F24_preprocessOneSingleImageForInference(image_path, transform)
        probs        = P07_F25_runForwardPassAndGetSoftmaxProbabilities(model, tensor, device)
        label, conf  = P07_F26_convertSoftmaxOutputToLabelAndConfidenceScore(probs, idx_to_class)

        taxon    = P03_F10_lookUpTaxonomyForThisLabelInCachedMaster(label)
        enriched = P03_F11_buildEnrichedPredictionDictionaryFromResult(label, conf, taxon)

        # for known_species: compare predicted label to true folder name
        # for excluded_species: no ground truth exists — mark N/A
        is_correct = str(label == folder_name) if category == "known_species" else "N/A"

        return {
            "image_filename":  os.path.basename(image_path),
            "category":        category,
            "true_folder":     folder_name,
            "predicted_label": enriched["internal_label"],
            "confidence":      round(enriched["confidence"] * 100, 2),
            "taxonomy_status": enriched["taxonomy_status"],
            "scientific_name": enriched["scientific_name"] or "",
            "is_correct":      is_correct,
        }

    except Exception as e:
        # return a placeholder row so the entire batch never crashes on one bad image
        WARN_LOGGER("P06", f"failed on {os.path.basename(image_path)}: {e}")
        return {
            "image_filename":  os.path.basename(image_path),
            "category":        category,
            "true_folder":     folder_name,
            "predicted_label": "ERROR",
            "confidence":      0.0,
            "taxonomy_status": "error",
            "scientific_name": "",
            "is_correct":      "ERROR",
        }


# FUNCTION 21
def P06_F21_runBulkInferenceAndCollectAllRows(
    image_records: list,
    model,
    transform,
    idx_to_class:  dict,
    device,
) -> list:
    # iterates over every image record, calls F20, collects result rows
    # logs progress every 500 images so large runs stay visible in the console

    rows  = []
    total = len(image_records)

    for i, rec in enumerate(image_records, start=1):
        row = P06_F20_predictOneImageAndReturnResultRow(
            image_path=rec["image_path"],
            category=rec["category"],
            folder_name=rec["folder_name"],
            model=model,
            transform=transform,
            idx_to_class=idx_to_class,
            device=device,
        )
        rows.append(row)

        if i % 500 == 0 or i == total:
            LOGGER("P06", f"  predicted {i}/{total}")

    return rows


# FUNCTION 22
def P06_F22_saveBulkPredictionResultsToCsvAndPrintSummary(rows: list) -> None:
    # writes all rows to CSV then logs a human-readable summary to the console

    os.makedirs(M07_P06_OUTPUT_BULK_PREDICT_FOLDER, exist_ok=True)

    # ── write CSV ──────────────────────────────────────────────────────────────
    fieldnames = [
        "image_filename", "category", "true_folder",
        "predicted_label", "confidence", "taxonomy_status",
        "scientific_name", "is_correct",
    ]
    with open(M07_P06_OUTPUT_BULK_PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER("P06", f"predictions CSV saved -> {M07_P06_OUTPUT_BULK_PREDICTIONS_CSV}")

    # ── compute summary stats ──────────────────────────────────────────────────
    known_rows    = [r for r in rows if r["category"] == "known_species"]
    correct_rows  = [r for r in known_rows if r["is_correct"] == "True"]
    excluded_rows = [r for r in rows if r["category"] == "excluded_species"]
    verified_count  = sum(1 for r in rows if r["taxonomy_status"] == "verified")
    low_conf_count  = sum(1 for r in rows if r["taxonomy_status"] == "low_confidence")
    known_acc = (len(correct_rows) / len(known_rows) * 100) if known_rows else 0.0

    excluded_pred_counts = Counter(r["predicted_label"] for r in excluded_rows)
    top_excluded = excluded_pred_counts.most_common(1)[0] if excluded_pred_counts else ("N/A", 0)

    LOGGER("P06", "─── bulk inference summary ─────────────────────────────────")
    LOGGER("P06", f"  total images predicted        : {len(rows)}")
    LOGGER("P06", f"  known species images          : {len(known_rows)}")
    LOGGER("P06", f"    correct predictions         : {len(correct_rows)}")
    LOGGER("P06", f"    accuracy on unseen known    : {known_acc:.2f}%")
    LOGGER("P06", f"  excluded species images       : {len(excluded_rows)}")
    LOGGER("P06", f"    top predicted class         : {top_excluded[0]} ({top_excluded[1]} times)")
    LOGGER("P06", f"  verified (taxonomy OK)        : {verified_count}")
    LOGGER("P06", f"  low confidence (no taxonomy)  : {low_conf_count}")
    LOGGER("P06", "────────────────────────────────────────────────────────────")

    # per-class breakdown for known species
    LOGGER("P06", "  per-class accuracy (known species, unseen images):")
    per_class_correct = Counter(r["true_folder"] for r in correct_rows)
    per_class_total   = Counter(r["true_folder"] for r in known_rows)
    for species in sorted(per_class_total.keys()):
        c   = per_class_correct.get(species, 0)
        t   = per_class_total[species]
        pct = c / t * 100 if t else 0.0
        LOGGER("P06", f"    {species:<42} {c:>5}/{t:<5}  {pct:.1f}%")

    # top-3 predicted class per excluded species folder
    if excluded_rows:
        LOGGER("P06", "  excluded species — top-3 model predictions per folder:")
        per_folder = {}
        for r in excluded_rows:
            per_folder.setdefault(r["true_folder"], []).append(r["predicted_label"])
        for folder in sorted(per_folder.keys()):
            top3    = Counter(per_folder[folder]).most_common(3)
            top_str = ", ".join(f"{label}({cnt})" for label, cnt in top3)
            LOGGER("P06", f"    {folder}: {top_str}")


# MAIN
if __name__ == "__main__":
    LOGGER("P06", "=== M07_P06 — bulk predict on remaining Fish4Knowledge images ===")
    LOGGER("P06", f"input folder: {M01_P05_OUTPUT_REMAINING_FOLDER}")
    LOGGER("P06", "NOTE: known_species alone has ~25 000 images — expect 20-40 min on GTX 1650")
    LOGGER("P06", "NOTE: excluded_species images will be forced into one of the 10 trained classes")

    LOGGER("P06", "step 1 — loading model, transform, class map...")
    model, transform, idx_to_class, device = P06_F18_loadModelAndTransformForBulkInference()

    LOGGER("P06", "step 2 — collecting all image paths from M01_P05 output...")
    image_records = P06_F19_collectAllImagePathsFromRemainingFolder()

    LOGGER("P06", "step 3 — running inference on all images...")
    rows = P06_F21_runBulkInferenceAndCollectAllRows(image_records, model, transform, idx_to_class, device)

    LOGGER("P06", "step 4 — saving CSV and printing summary...")
    P06_F22_saveBulkPredictionResultsToCsvAndPrintSummary(rows)

    LOGGER("P06", f"=== M07_P06 done — results at {M07_P06_OUTPUT_BULK_PREDICTIONS_CSV} ===")