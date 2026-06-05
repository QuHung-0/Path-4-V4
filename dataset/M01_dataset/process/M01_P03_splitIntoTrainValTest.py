# ========= FILE NAME: M01_P03_splitIntoTrainValTest.py =========
# FILE ROLE: Takes balanced 200-per-species folder and splits each species into 140/30/30
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P03_F09 → P03_F12

# import OS utilities for file and folder operations
import os

# import random module for shuffling data
import random

# import shutil for copying files
import shutil

# import paths, settings, and logging utilities
from dataset.M01_dataset.input.M01_I01_inputPaths import (
    M01_P02_OUTPUT_BALANCED_FOLDER,   # input: balanced dataset (200 images per species)

    M01_P03_OUTPUT_TRAIN_FOLDER,      # output folder for training set
    M01_P03_OUTPUT_VAL_FOLDER,        # output folder for validation set
    M01_P03_OUTPUT_TEST_FOLDER,       # output folder for test set

    M01_P03_TRAIN_COUNT,              # number of training images per species (140)
    M01_P03_VAL_COUNT,                # number of validation images per species (30)
    M01_P03_TEST_COUNT,               # number of test images per species (30)

    RANDOM_SEED,                     # seed for reproducibility

    LOGGER,                           # logging function
    WARN_LOGGER                       # warning logging function
)


# FUNCTION 9
def P03_F09_loadAllImagePathsGroupedBySpecies(balanced_folder: str) -> dict:
    # reads balanced folder and returns {species_name: [list of full image paths]}

    grouped = {}  # dictionary to store species → list of image paths

    # iterate through species folders in sorted order (for consistency)
    for species in sorted(os.listdir(balanced_folder)):

        species_path = os.path.join(balanced_folder, species)

        # skip non-directory items
        if not os.path.isdir(species_path):
            continue

        # collect full paths of all image files
        images = [
            os.path.join(species_path, f)
            for f in os.listdir(species_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # store list in dictionary
        grouped[species] = images

        # log how many images were loaded for this species
        LOGGER("P03", f"  {species}: {len(images)} images loaded")

    return grouped


# FUNCTION 10
def P03_F10_shuffleAndSplitEachSpeciesIntoThreeGroups(
    grouped: dict, train_n: int, val_n: int, test_n: int, seed: int
) -> dict:
    # shuffles each species list with fixed seed, then slices into three groups

    # set random seed for reproducibility
    random.seed(seed)

    splits = {}  # dictionary to store final splits

    # iterate through each species and its image list
    for species, images in grouped.items():

        # create a copy of the list so original order is not modified
        shuffled = images.copy()

        # shuffle list randomly
        random.shuffle(shuffled)

        # slice list into train/val/test segments
        splits[species] = {
            "train": shuffled[:train_n],                                   # first 140 images
            "val":   shuffled[train_n:train_n + val_n],                    # next 30 images
            "test":  shuffled[train_n + val_n:train_n + val_n + test_n],   # final 30 images
        }

    return splits


# FUNCTION 11
def P03_F11_copyEachGroupIntoTrainValTestSubfolders(
    splits: dict, train_folder: str, val_folder: str, test_folder: str
) -> None:
    # copies files from split lists into the right output subfolders

    # map split names to their destination folders
    folder_map = {
        "train": train_folder,
        "val": val_folder,
        "test": test_folder
    }

    # iterate through each species
    for species, groups in splits.items():

        # iterate through each split (train/val/test)
        for split_name, image_paths in groups.items():

            # determine destination folder for this split and species
            dst = os.path.join(folder_map[split_name], species)

            # create destination directory if it does not exist
            os.makedirs(dst, exist_ok=True)

            # copy each image file into destination
            for src_path in image_paths:
                shutil.copy(
                    src_path,                              # source file path
                    os.path.join(dst, os.path.basename(src_path))  # destination path
                )

        # log summary for this species
        LOGGER(
            "P03",
            f"  {species}: train={len(groups['train'])} val={len(groups['val'])} test={len(groups['test'])}"
        )


# FUNCTION 12
def P03_F12_verifySplitHasCorrectCountsInAllThreeFolders(
    train_folder: str, val_folder: str, test_folder: str,
    expected_train: int, expected_val: int, expected_test: int
) -> bool:
    # checks that every species in every split has the right number of images

    all_ok = True  # assume everything is correct

    # map each split to its folder and expected count
    folder_map = {
        "train": (train_folder, expected_train),
        "val":   (val_folder,   expected_val),
        "test":  (test_folder,  expected_test),
    }

    # iterate through each split
    for split_name, (folder, expected) in folder_map.items():

        # iterate through species folders
        for species in sorted(os.listdir(folder)):

            path = os.path.join(folder, species)

            # skip non-directories
            if not os.path.isdir(path):
                continue

            # count images in this folder
            actual = len([f for f in os.listdir(path)
                          if f.lower().endswith((".png", ".jpg", ".jpeg"))])

            # compare with expected count
            if actual != expected:
                WARN_LOGGER("P03", f"{split_name}/{species}: {actual} images, expected {expected}")
                all_ok = False

    return all_ok


# MAIN
if __name__ == "__main__":

    LOGGER("P03", "=== M01_P03 — splitting into train / val / test ===")

    # STEP 1 — load balanced dataset
    LOGGER("P03", "step 1 — loading balanced folder...")

    grouped = P03_F09_loadAllImagePathsGroupedBySpecies(
        M01_P02_OUTPUT_BALANCED_FOLDER
    )

    LOGGER("P03", f"{len(grouped)} species loaded")


    # STEP 2 — split dataset
    LOGGER(
        "P03",
        f"step 2 — splitting {M01_P03_TRAIN_COUNT}/{M01_P03_VAL_COUNT}/{M01_P03_TEST_COUNT} (seed={RANDOM_SEED})..."
    )

    splits = P03_F10_shuffleAndSplitEachSpeciesIntoThreeGroups(
        grouped,
        M01_P03_TRAIN_COUNT,
        M01_P03_VAL_COUNT,
        M01_P03_TEST_COUNT,
        RANDOM_SEED
    )


    # STEP 3 — copy files
    LOGGER("P03", "step 3 — copying files into output folders...")

    P03_F11_copyEachGroupIntoTrainValTestSubfolders(
        splits,
        M01_P03_OUTPUT_TRAIN_FOLDER,
        M01_P03_OUTPUT_VAL_FOLDER,
        M01_P03_OUTPUT_TEST_FOLDER
    )


    # STEP 4 — verify counts
    LOGGER("P03", "step 4 — verifying all counts...")

    ok = P03_F12_verifySplitHasCorrectCountsInAllThreeFolders(
        M01_P03_OUTPUT_TRAIN_FOLDER,
        M01_P03_OUTPUT_VAL_FOLDER,
        M01_P03_OUTPUT_TEST_FOLDER,
        M01_P03_TRAIN_COUNT,
        M01_P03_VAL_COUNT,
        M01_P03_TEST_COUNT
    )

    if ok:
        LOGGER("P03", "all split counts verified ✓")
    else:
        WARN_LOGGER("P03", "some counts are wrong — check warnings above")


    LOGGER("P03", "=== M01_P03 done ===")
    LOGGER("P03", f"train → {M01_P03_OUTPUT_TRAIN_FOLDER}")
    LOGGER("P03", f"val   → {M01_P03_OUTPUT_VAL_FOLDER}")
    LOGGER("P03", f"test  → {M01_P03_OUTPUT_TEST_FOLDER}")