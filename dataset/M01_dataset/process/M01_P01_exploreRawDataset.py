# ========= FILE NAME: M01_P01_exploreRawDataset.py =========
# FILE ROLE: Counts images per species, finds size extremes, checks for corrupt files, saves report
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P01_F01 → P01_F04

# import system-related utilities (not explicitly used later, but often included for consistency/debugging)
import sys, os

# import CSV module for writing structured tabular data to file
import csv

# import Image class from PIL (Python Imaging Library) to open and inspect images
from PIL import Image

# import paths and logging utilities from the M01 input configuration file
from dataset.M01_dataset.input.M01_I01_inputPaths import (
    M01_INPUT_RAW_FISH_IMAGE_FOLDER,   # root folder containing raw dataset (species folders inside)

    M01_P01_OUTPUT_EXPLORATION_CSV,    # output CSV file path for saving dataset exploration results
    M01_OUTPUT_REPORTS_FOLDER,         # general reports folder (not directly used here but available)

    LOGGER,                           # function to print normal log messages with timestamp
    WARN_LOGGER                       # function to print warning messages with timestamp
)


# FUNCTION 1
def P01_F01_countImagesInEachSpeciesFolder(raw_folder: str) -> dict:
    # walks raw folder and returns {species_name: image_count}

    counts = {}  # initialize empty dictionary to store results

    # loop through every item (file or folder) inside raw_folder, sorted alphabetically
    for species in sorted(os.listdir(raw_folder)):

        # build full path to the species folder
        species_path = os.path.join(raw_folder, species)

        # skip anything that is not a directory (ignore stray files)
        if not os.path.isdir(species_path):
            continue

        # list all files in this species folder and filter only image files
        images = [f for f in os.listdir(species_path)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # store the number of images for this species
        counts[species] = len(images)

    # return dictionary like: {"species_A": 250, "species_B": 180, ...}
    return counts



# FUNCTION 2
def P01_F02_findSmallestAndLargestImageSizeInDataset(raw_folder: str) -> dict:
    # opens every image and tracks the min and max width and height seen

    # initialize minimum width/height with infinity so any real image will be smaller
    min_w, min_h = float("inf"), float("inf")

    # initialize maximum width/height with 0 so any real image will be larger
    max_w, max_h = 0, 0

    # store file paths of extreme cases
    min_file = max_file = ""

    # counter for how many images were successfully checked
    checked = 0

    # iterate through each species folder
    for species in os.listdir(raw_folder):

        species_path = os.path.join(raw_folder, species)

        # skip non-directories
        if not os.path.isdir(species_path):
            continue

        # iterate through files in species folder
        for fname in os.listdir(species_path):

            # skip non-image files
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            # build full file path
            fpath = os.path.join(species_path, fname)

            try:
                # open image safely using context manager (auto closes file)
                with Image.open(fpath) as img:

                    # get image dimensions (width, height)
                    w, h = img.size

                    # check if this image is smaller than current minimum
                    if w < min_w or h < min_h:
                        min_w, min_h, min_file = w, h, fpath

                    # check if this image is larger than current maximum
                    if w > max_w or h > max_h:
                        max_w, max_h, max_file = w, h, fpath

                    # increment count of successfully processed images
                    checked += 1

            except Exception:
                # if image cannot be opened (corrupt, unsupported, etc.), silently skip
                pass

    # return all collected statistics in a dictionary
    return {
        "min_width":  min_w,     # smallest width seen
        "min_height": min_h,     # smallest height seen
        "min_file":   min_file,  # file path of smallest image

        "max_width":  max_w,     # largest width seen
        "max_height": max_h,     # largest height seen
        "max_file":   max_file,  # file path of largest image

        "total_checked": checked,  # number of images successfully analyzed
    }


# FUNCTION 3
def P01_F03_scanForCorruptOrUnreadableImages(raw_folder: str) -> list:
    # tries to open every image and records any that fail to open

    bad_files = []  # list to store info about problematic images

    # iterate through species folders
    for species in os.listdir(raw_folder):

        species_path = os.path.join(raw_folder, species)

        # skip non-directories
        if not os.path.isdir(species_path):
            continue

        # iterate through files
        for fname in os.listdir(species_path):

            # skip non-image files
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            # full path to image
            fpath = os.path.join(species_path, fname)

            try:
                # attempt to open and verify image integrity
                with Image.open(fpath) as img:
                    img.verify()  # checks if file is a valid image without fully decoding

            except Exception as e:
                # if any error occurs, record the file and the error message
                bad_files.append({
                    "file": fpath,
                    "error": str(e)
                })

    # return list of problematic files (can be empty if dataset is clean)
    return bad_files



# FUNCTION 4
def P01_F04_saveExplorationResultsToCSV(counts: dict, output_csv: str) -> None:
    # writes one row per species with image count to the reports folder

    # ensure output directory exists (create if not)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # open CSV file for writing
    with open(output_csv, "w", newline="", encoding="utf-8") as f:

        # create a CSV writer with defined column names
        writer = csv.DictWriter(f, fieldnames=["species_folder", "image_count"])

        # write header row
        writer.writeheader()

        # write one row per species
        for species, count in counts.items():
            writer.writerow({
                "species_folder": species,
                "image_count": count
            })

    # log that file has been saved
    LOGGER("P01", f"exploration CSV saved → {output_csv}")



# MAIN
if __name__ == "__main__":

    # log start of script
    LOGGER("P01", "=== M01_P01 — exploring raw dataset ===")

    # log location of raw dataset
    LOGGER("P01", f"raw folder: {M01_INPUT_RAW_FISH_IMAGE_FOLDER}")


    # STEP 1 — count images
    LOGGER("P01", "step 1 — counting images per species...")

    # call function to count images
    counts = P01_F01_countImagesInEachSpeciesFolder(M01_INPUT_RAW_FISH_IMAGE_FOLDER)

    # compute total number of images across all species
    total_images  = sum(counts.values())

    # compute total number of species
    total_species = len(counts)

    LOGGER("P01", f"found {total_species} species, {total_images} images total")


    # print detailed breakdown to console
    print("\n--- IMAGE COUNT PER SPECIES ---")

    for species, count in counts.items():

        # flag species with fewer than 200 images
        flag = " ← below 200" if count < 200 else ""

        print(f"  {species}: {count}{flag}")

    print(f"\n  TOTAL: {total_images} images across {total_species} species")


    # STEP 2 — analyze image sizes
    LOGGER("P01", "step 2 — finding smallest and largest images...")

    sizes = P01_F02_findSmallestAndLargestImageSizeInDataset(
        M01_INPUT_RAW_FISH_IMAGE_FOLDER
    )

    LOGGER("P01", f"smallest: {sizes['min_width']}×{sizes['min_height']} px")
    LOGGER("P01", f"largest:  {sizes['max_width']}×{sizes['max_height']} px")
    LOGGER("P01", f"checked:  {sizes['total_checked']} images")


    # STEP 3 — check for corrupt images
    LOGGER("P01", "step 3 — scanning for corrupt images...")

    bad = P01_F03_scanForCorruptOrUnreadableImages(
        M01_INPUT_RAW_FISH_IMAGE_FOLDER
    )

    if bad:
        # if any corrupt images found, print warnings
        WARN_LOGGER("P01", f"{len(bad)} corrupt files found:")

        for b in bad:
            WARN_LOGGER("P01", f"  {b['file']} — {b['error']}")

    else:
        # dataset is clean
        LOGGER("P01", "no corrupt images found — dataset is clean")


    # STEP 4 — save CSV report
    LOGGER("P01", "step 4 — saving exploration report...")

    P01_F04_saveExplorationResultsToCSV(
        counts,
        M01_P01_OUTPUT_EXPLORATION_CSV
    )

    # final log
    LOGGER("P01", "=== M01_P01 done — check reports/ for the CSV ===")