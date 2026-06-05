# ========= FILE NAME: M02_P01_computeNormalizationStatsFromTrainSet.py =========
# FILE ROLE: Measures average colour and colour variation across all training images, saves result
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P01_F01 → P01_F03

# import OS utilities
import os

# import JSON module for saving results
import json

# import NumPy for numerical operations
import numpy as np

# import PIL for image loading
from PIL import Image

# import paths, settings, and logger
from dataset.M02_preprocessing.input.M02_I01_inputPaths   import (
    M02_P01_P02_INPUT_TRAIN_FOLDER,   # training dataset path
    M02_P01_OUTPUT_NORM_STATS_JSON,   # output JSON file path

    M02_P01_P02_INPUT_IMAGE_SIZE,     # resize dimension (e.g., 224)

    LOGGER                            # logging function
)


# FUNCTION 1
def P01_F01_collectAllImagePathsFromTrainFolder(train_folder: str) -> list:
    # walks every species subfolder and returns a flat list of all image file paths

    all_paths = []  # list to store full file paths

    # iterate through species folders
    for species in sorted(os.listdir(train_folder)):

        species_path = os.path.join(train_folder, species)

        # skip non-directories
        if not os.path.isdir(species_path):
            continue

        # iterate through files inside species folder
        for fname in os.listdir(species_path):

            # keep only image files
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):

                # append full path to list
                all_paths.append(os.path.join(species_path, fname))

    LOGGER("P01", f"found {len(all_paths)} training images to measure")

    return all_paths


# FUNCTION 2
def P01_F02_computePerChannelMeanAndStdDeviation(image_paths: list, target_size: int) -> tuple:
    # opens every image, resizes to target_size, accumulates pixel sums, computes mean and std

    # initialize sum of RGB values
    sum_rgb = np.zeros(3, dtype=np.float64)

    # initialize sum of squared RGB values (needed for std calculation)
    sum_sq_rgb = np.zeros(3, dtype=np.float64)

    # total number of pixels processed
    pixel_count = 0

    total = len(image_paths)  # total number of images

    # loop through all images
    for i, path in enumerate(image_paths):

        # print progress every 200 images
        if i % 200 == 0:
            LOGGER("P01", f"  processing image {i+1}/{total}...")

        # open image, convert to RGB (ensures 3 channels), resize to fixed size
        img = Image.open(path).convert("RGB").resize((target_size, target_size))

        # convert image to NumPy array and scale pixel values to [0,1]
        arr = np.array(img, dtype=np.float64) / 255.0  # shape: (H, W, 3)

        # sum RGB values across all pixels
        sum_rgb += arr.sum(axis=(0, 1))

        # sum squared RGB values (for variance calculation)
        sum_sq_rgb += (arr ** 2).sum(axis=(0, 1))

        # update pixel count
        pixel_count += target_size * target_size

    # compute mean: E[X]
    mean = sum_rgb / pixel_count

    # compute standard deviation: sqrt(E[X²] - (E[X])²)
    std  = np.sqrt((sum_sq_rgb / pixel_count) - (mean ** 2))

    # log results
    LOGGER("P01", f"mean (RGB): [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    LOGGER("P01", f"std  (RGB): [{std[0]:.6f},  {std[1]:.6f},  {std[2]:.6f}]")

    return mean.tolist(), std.tolist()


# FUNCTION 3
def P01_F03_saveNormalizationStatsToJsonFile(mean: list, std: list, output_path: str) -> None:
    # writes mean and std to a JSON file

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # create dictionary
    stats = {
        "mean": mean,
        "std": std
    }

    # write JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    LOGGER("P01", f"norm stats saved → {output_path}")


# MAIN
if __name__ == "__main__":

    LOGGER("P01", "=== M02_P01 — computing normalization stats from training set ===")

    LOGGER("P01", f"train folder: {M02_P01_P02_INPUT_TRAIN_FOLDER}")
    LOGGER("P01", f"image size used for measurement: {M02_P01_P02_INPUT_IMAGE_SIZE}×{M02_P01_P02_INPUT_IMAGE_SIZE}")

    # STEP 1
    LOGGER("P01", "step 1 — collecting all training image paths...")

    paths = P01_F01_collectAllImagePathsFromTrainFolder(
        M02_P01_P02_INPUT_TRAIN_FOLDER
    )

    # STEP 2
    LOGGER("P01", "step 2 — computing mean and std (this takes ~1 minute)...")

    mean, std = P01_F02_computePerChannelMeanAndStdDeviation(
        paths,
        M02_P01_P02_INPUT_IMAGE_SIZE
    )

    # STEP 3
    LOGGER("P01", "step 3 — saving stats to file...")

    P01_F03_saveNormalizationStatsToJsonFile(
        mean,
        std,
        M02_P01_OUTPUT_NORM_STATS_JSON
    )

    LOGGER("P01", "=== M02_P01 done ===")
    LOGGER("P01", "these numbers will be used to normalize every image the model ever sees")