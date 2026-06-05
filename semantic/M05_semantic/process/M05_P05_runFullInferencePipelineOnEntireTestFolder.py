# ========= FILE NAME: M05_P05_runFullInferencePipelineOnEntireTestFolder.py =========
# FILE ROLE: Predicts every test image, enriches with taxonomy, writes one JSON per image
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P05_F15 → P05_F17

# import OS utilities for folder traversal
import os

# import PyTorch for model execution
import torch

# import shared M05 inputs, outputs, and logger
from semantic.M05_semantic.input.M05_I01_inputPaths import (
    M05_P05_INPUT_TEST_FOLDER,        # test folder from M01
    M05_P05_INPUT_BEST_MODEL_PTH,     # trained model weights
    M05_P05_INPUT_CLASS_TO_IDX_JSON,  # class mapping JSON
    M05_P05_INPUT_NORM_STATS_JSON,    # normalization stats JSON

    M05_P05_OUTPUT_INFERENCE_RESULTS_FOLDER,  # folder for per-image JSON results

    M05_P05_INPUT_IMAGE_SIZE,         # image size for inference transform

    M05_P05_OUTPUT_RUN_TXT,           # log text file for this run
    M05_OUTPUT_FOLDER,                # M05 output root

    LOGGER,                           # logger
)

# import enrichment helpers
from semantic.M05_semantic.process.M05_P03_enrichOnePredictionWithTaxonomyFromCache import (
    P03_F10_lookUpTaxonomyForThisLabelInCachedMaster,     # taxonomy lookup
    P03_F11_buildEnrichedPredictionDictionaryFromResult,   # build enriched prediction dict
)

# import JSON record helpers
from semantic.M05_semantic.process.M05_P04_buildAndWriteJsonRecordForOneInference import (
    P04_F12_generateContentHashFromImageFileBytes,         # image hash for ID
    P04_F13_assembleCompleteJsonRecordFromAllFields,       # build final record
    P04_F14_writeJsonRecordToResultsFolder,               # save JSON file
)

# import model loading and inference helpers from M03
from model_weights.M03_model.process.M03_P07_loadModelAndPredictOneSingleImage import (
    P07_F23_loadTrainedModelWeightsFromFile,              # load trained model
    P07_F24_preprocessOneSingleImageForInference,         # preprocess image
    P07_F25_runForwardPassAndGetSoftmaxProbabilities,     # run model + softmax
    P07_F26_convertSoftmaxOutputToLabelAndConfidenceScore,# map probs to label/confidence
)

# import class mapping loader from M03
from model_weights.M03_model.process.M03_P03_buildDataloadersForTrainingAndValidation import (
    P03_F11_loadClassToIndexMappingFromJsonFile,          # load idx→class mapping
)

# import normalization and inference transform helpers from M02
from dataset.M02_preprocessing.process.M02_P02_defineAllTransformPipelines import (
    P02_F04_loadNormalizationStatsFromJsonFile,           # load mean/std
    P02_F07_buildInferenceTransformForSingleImage,        # build inference transform
)

# FUNCTION 15
def P05_F15_walkTestFolderAndListAllImagePaths(test_folder: str) -> list:
    # recursively finds every image file in the test folder
    image_paths = []
    for species in sorted(os.listdir(test_folder)):
        species_path = os.path.join(test_folder, species)
        if not os.path.isdir(species_path):
            continue
        for fname in sorted(os.listdir(species_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(species_path, fname))
    LOGGER("P05", f"found {len(image_paths)} images in test folder")
    return image_paths

# FUNCTION 16
def P05_F16_predictEnrichAndWriteJsonForOneImage(
    image_path: str, model, transform, idx_to_class: dict, device: torch.device
) -> dict:
    # runs the full pipeline on one image: predict → enrich → build record
    tensor = P07_F24_preprocessOneSingleImageForInference(image_path, transform)
    probs = P07_F25_runForwardPassAndGetSoftmaxProbabilities(model, tensor, device)
    label, conf = P07_F26_convertSoftmaxOutputToLabelAndConfidenceScore(probs, idx_to_class)
    taxon = P03_F10_lookUpTaxonomyForThisLabelInCachedMaster(label)
    enriched = P03_F11_buildEnrichedPredictionDictionaryFromResult(label, conf, taxon)
    image_id = P04_F12_generateContentHashFromImageFileBytes(image_path)
    record = P04_F13_assembleCompleteJsonRecordFromAllFields(image_id, image_path, enriched)
    return record

# FUNCTION 17
def P05_F17_printProgressAndWriteEachRecordToResultsFolder(
    image_paths: list, model, transform, idx_to_class: dict, device: torch.device
) -> tuple:
    # processes all images, writes each JSON, prints progress every 50 images
    os.makedirs(M05_P05_OUTPUT_INFERENCE_RESULTS_FOLDER, exist_ok=True)
    os.makedirs(M05_OUTPUT_FOLDER, exist_ok=True)

    verified_count = 0  # count of records with verified taxonomy
    low_conf_count = 0  # count of records below confidence threshold
    total = len(image_paths)

    with open(M05_P05_OUTPUT_RUN_TXT, "w", encoding="utf-8") as log_file:
        log_file.write("image_filename,internal_label,confidence,taxonomy_status,json_file\n")

        for i, image_path in enumerate(image_paths):
            record = P05_F16_predictEnrichAndWriteJsonForOneImage(
                image_path, model, transform, idx_to_class, device
            )
            json_path = P04_F14_writeJsonRecordToResultsFolder(record, M05_P05_OUTPUT_INFERENCE_RESULTS_FOLDER)

            status = record["taxonomy_status"]
            if status == "verified":
                verified_count += 1
            else:
                low_conf_count += 1

            log_file.write(
                f"{record['image_filename']},"
                f"{record['internal_label']},"
                f"{record['confidence']:.4f},"
                f"{status},"
                f"{os.path.basename(json_path)}\n"
            )

            if (i + 1) % 50 == 0 or (i + 1) == total:
                LOGGER("P05", f"  processed {i+1}/{total} — verified: {verified_count}  low_conf: {low_conf_count}")

    return verified_count, low_conf_count

# MAIN
if __name__ == "__main__":
    LOGGER("P05", "=== M05_P05 — full inference pipeline on test folder ===")

    # choose compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER("P05", f"device: {device}")

    LOGGER("P05", "step 1 — loading model, transform, class map...")

    # load normalization stats and build inference transform
    mean, std = P02_F04_loadNormalizationStatsFromJsonFile(M05_P05_INPUT_NORM_STATS_JSON)
    transform = P02_F07_buildInferenceTransformForSingleImage(mean, std, M05_P05_INPUT_IMAGE_SIZE)

    # load index→class mapping and trained model
    idx_to_class = P03_F11_loadClassToIndexMappingFromJsonFile(M05_P05_INPUT_CLASS_TO_IDX_JSON)
    model = P07_F23_loadTrainedModelWeightsFromFile(M05_P05_INPUT_BEST_MODEL_PTH, device)

    LOGGER("P05", "step 2 — collecting all test image paths...")
    image_paths = P05_F15_walkTestFolderAndListAllImagePaths(M05_P05_INPUT_TEST_FOLDER)

    LOGGER("P05", "step 3 — running predict → enrich → save JSON for every image...")
    verified, low_conf = P05_F17_printProgressAndWriteEachRecordToResultsFolder(
        image_paths, model, transform, idx_to_class, device
    )

    total = len(image_paths)
    LOGGER("P05", "─── pipeline complete ──────────────────────────────")
    LOGGER("P05", f"  total images processed:  {total}")
    LOGGER("P05", f"  verified (taxonomy OK):  {verified}")
    LOGGER("P05", f"  low confidence (no tax): {low_conf}")
    LOGGER("P05", f"  JSON files written to:   {M05_P05_OUTPUT_INFERENCE_RESULTS_FOLDER}")
    LOGGER("P05", f"  run log saved to:        {M05_P05_OUTPUT_RUN_TXT}")
    LOGGER("P05", "────────────────────────────────────────────────────")
    LOGGER("P05", "=== M05_P05 done ===")