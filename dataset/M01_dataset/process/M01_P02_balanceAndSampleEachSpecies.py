# ========= FILE NAME: M01_P02_balanceAndSampleEachSpecies.py =========
# FILE ROLE: Keeps only species with enough images, copies exactly 200 random images each
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P02_F05 → P02_F08

# import system and path utilities
import sys, os

# import JSON module to read mapping file
import json

# import random module for sampling images
import random

# import shutil to copy files from one folder to another
import shutil

# import CSV module for writing report
import csv

# import paths, settings, and logging utilities from input config
from dataset.M01_dataset.input.M01_I01_inputPaths  import (
    M01_INPUT_RAW_FISH_IMAGE_FOLDER,     # raw dataset root folder
    M01_P02_INPUT_SPECIES_MAP,           # JSON file mapping folder IDs → species names

    M01_P02_OUTPUT_BALANCED_FOLDER,      # output folder for balanced dataset
    M01_OUTPUT_REPORTS_FOLDER,           # reports folder

    M01_P02_MIN_IMAGES_PER_SPECIES,      # minimum images required to keep a species
    M01_P02_SAMPLE_SIZE_PER_SPECIES,     # number of images to sample per species
    RANDOM_SEED,                        # seed for reproducibility

    LOGGER,                             # normal logging function
    WARN_LOGGER                         # warning logging function
)


# FUNCTION 5
def P02_F05_loadSpeciesMapAndKeepOnlyQualifyingSpecies(
    raw_folder: str, map_json: str, min_count: int
) -> dict:
    # loads the fish_XX → species_name map, counts images, drops species below min_count

    # open and read JSON mapping file
    with open(map_json, "r", encoding="utf-8") as f:
        label_map = json.load(f)   # example: {"fish_01": "Dascyllus_reticulatus", ...}

    qualified = {}  # dictionary to store species that meet requirements

    # iterate through each folder ID and its mapped species name
    for folder_id, species_name in label_map.items():

        # construct path to that species folder in raw dataset
        folder_path = os.path.join(raw_folder, folder_id)

        # if folder does not exist, log warning and skip
        if not os.path.isdir(folder_path):
            WARN_LOGGER("P02", f"{folder_id} not found in raw folder — skipping")
            continue

        # list all image files in the folder
        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        count = len(images)  # number of images in this species

        # if species does not meet minimum requirement, drop it
        if count < min_count:
            WARN_LOGGER("P02", f"{folder_id} ({species_name}) has only {count} images — dropped")
            continue

        # otherwise, keep it and store its info
        qualified[folder_id] = {
            "species_name": species_name,
            "image_count": count
        }

    # log how many species passed the filter
    LOGGER("P02", f"{len(qualified)} species qualified (≥ {min_count} images each)")

    return qualified


# FUNCTION 6
def P02_F06_copyExactlyNRandomImagesIntoBalancedFolder(
    raw_folder: str, qualified: dict, n: int, seed: int, balanced_folder: str
) -> dict:
    # for each qualifying species, randomly picks n images and copies them

    # set random seed so sampling is reproducible
    random.seed(seed)

    report = {}  # store summary of what was copied

    # iterate through each qualified species
    for folder_id, info in qualified.items():

        species_name = info["species_name"]

        # source folder (raw dataset)
        src_folder   = os.path.join(raw_folder, folder_id)

        # destination folder (balanced dataset)
        dst_folder   = os.path.join(balanced_folder, species_name)

        # create destination folder if it does not exist
        os.makedirs(dst_folder, exist_ok=True)

        # list all images in source folder
        all_images = [f for f in os.listdir(src_folder)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # randomly select exactly n images (no replacement)
        chosen = random.sample(all_images, n)

        # copy each selected image to destination folder
        for fname in chosen:
            shutil.copy(
                os.path.join(src_folder, fname),         # source file path
                os.path.join(dst_folder, fname)          # destination file path
            )

        # log progress for this species
        LOGGER("P02", f"  {species_name}: {n} images copied from {folder_id}")

        # record in report
        report[species_name] = {
            "source_folder": folder_id,
            "copied": n
        }

    return report


# FUNCTION 7
def P02_F07_verifyBalancedFolderHasCorrectCountPerSpecies(
    balanced_folder: str, expected_count: int
) -> bool:
    # opens balanced folder and checks every species has exactly the right number of images

    all_ok = True  # assume everything is correct initially

    # iterate through each species folder in balanced dataset
    for species in os.listdir(balanced_folder):

        path = os.path.join(balanced_folder, species)

        # skip non-directory items
        if not os.path.isdir(path):
            continue

        # count number of image files
        actual = len([f for f in os.listdir(path)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        # compare with expected count
        if actual != expected_count:
            WARN_LOGGER("P02", f"{species} has {actual} images but expected {expected_count}")
            all_ok = False
        else:
            LOGGER("P02", f"  {species}: {actual} ✓")

    return all_ok


# FUNCTION 8
def P02_F08_saveBalancedSamplingReportToCSV(report: dict, reports_folder: str) -> None:
    # writes a summary of what was sampled from where

    # ensure reports folder exists
    os.makedirs(reports_folder, exist_ok=True)

    # define output CSV path
    out_path = os.path.join(reports_folder, "M01_P02_balanced_sampling_report.csv")

    # open file for writing
    with open(out_path, "w", newline="", encoding="utf-8") as f:

        # define CSV columns
        writer = csv.DictWriter(f, fieldnames=["species_name", "source_folder", "copied"])

        # write header row
        writer.writeheader()

        # write one row per species
        for species_name, info in report.items():
            writer.writerow({
                "species_name": species_name,
                **info   # expands dictionary (source_folder + copied)
            })

    # log completion
    LOGGER("P02", f"sampling report saved → {out_path}")


# MAIN
if __name__ == "__main__":

    LOGGER("P02", "=== M01_P02 — balancing and sampling dataset ===")

    # STEP 1 — filter species
    LOGGER("P02", "step 1 — loading species map and checking image counts...")

    qualified = P02_F05_loadSpeciesMapAndKeepOnlyQualifyingSpecies(
        M01_INPUT_RAW_FISH_IMAGE_FOLDER,
        M01_P02_INPUT_SPECIES_MAP,
        M01_P02_MIN_IMAGES_PER_SPECIES
    )

    # STEP 2 — sample images
    LOGGER("P02", f"step 2 — copying {M01_P02_SAMPLE_SIZE_PER_SPECIES} images per species (seed={RANDOM_SEED})...")

    report = P02_F06_copyExactlyNRandomImagesIntoBalancedFolder(
        M01_INPUT_RAW_FISH_IMAGE_FOLDER,
        qualified,
        M01_P02_SAMPLE_SIZE_PER_SPECIES,
        RANDOM_SEED,
        M01_P02_OUTPUT_BALANCED_FOLDER
    )

    # STEP 3 — verify counts
    LOGGER("P02", "step 3 — verifying balanced folder counts...")

    ok = P02_F07_verifyBalancedFolderHasCorrectCountPerSpecies(
        M01_P02_OUTPUT_BALANCED_FOLDER,
        M01_P02_SAMPLE_SIZE_PER_SPECIES
    )

    if ok:
        LOGGER("P02", "all species verified ✓")
    else:
        WARN_LOGGER("P02", "some species have wrong counts — check warnings above")

    # STEP 4 — save report
    LOGGER("P02", "step 4 — saving report...")

    P02_F08_saveBalancedSamplingReportToCSV(
        report,
        M01_OUTPUT_REPORTS_FOLDER
    )

    LOGGER("P02", f"=== M01_P02 done — balanced folder: {M01_P02_OUTPUT_BALANCED_FOLDER} ===")