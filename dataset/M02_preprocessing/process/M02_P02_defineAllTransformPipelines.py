# ========= FILE NAME: M02_P02_defineAllTransformPipelines.py =========
# FILE ROLE: Defines three image transform recipes — train (with augmentation), val/test, inference
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P02_F04 → P02_F07

# import OS utilities (not heavily used here)
import os

# import JSON module
import json

# import torchvision transforms (core image preprocessing tools)
from torchvision import transforms

# import PIL for image handling (used in sanity check)
from PIL import Image

# import paths, settings, and logger
from dataset.M02_preprocessing.input.M02_I01_inputPaths   import (
    M02_P01_P02_INPUT_TRAIN_FOLDER,
    M02_P01_OUTPUT_NORM_STATS_JSON,

    M02_P01_P02_INPUT_IMAGE_SIZE,

    LOGGER
)


# FUNCTION 4
def P02_F04_loadNormalizationStatsFromJsonFile(stats_json_path: str) -> tuple:
    # reads norm_stats.json and returns (mean, std)

    with open(stats_json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = stats["mean"]
    std  = stats["std"]

    LOGGER("P02", f"loaded mean: {[round(v,4) for v in mean]}")
    LOGGER("P02", f"loaded std:  {[round(v,4) for v in std]}")

    return mean, std


# FUNCTION 5
def P02_F05_buildTrainTransformWithAugmentation(mean: list, std: list, size: int):
    # builds training transform with augmentation

    transform = transforms.Compose([
        transforms.Resize((size, size)),                    # resize image to fixed size
        transforms.RandomHorizontalFlip(p=0.5),             # randomly flip horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # random color changes
        transforms.RandomRotation(degrees=15),              # random rotation
        transforms.ToTensor(),                              # convert image to tensor (C,H,W)
        transforms.Normalize(mean=mean, std=std),           # normalize using dataset stats
    ])

    LOGGER("P02", "train transform built: Resize → HFlip → ColorJitter → Rotation → ToTensor → Normalize")

    return transform


# FUNCTION 6
def P02_F06_buildValAndTestTransformNoAugmentation(mean: list, std: list, size: int):
    # builds validation/test transform (no randomness)

    transform = transforms.Compose([
        transforms.Resize((size, size)),    # resize
        transforms.ToTensor(),              # convert to tensor
        transforms.Normalize(mean=mean, std=std),  # normalize
    ])

    LOGGER("P02", "val/test transform built: Resize → ToTensor → Normalize (no augmentation)")

    return transform


# FUNCTION 7
def P02_F07_buildInferenceTransformForSingleImage(mean: list, std: list, size: int):
    # same as val/test, used for single-image prediction

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    LOGGER("P02", "inference transform built: Resize → ToTensor → Normalize")

    return transform


# MAIN
if __name__ == "__main__":

    LOGGER("P02", "=== M02_P02 — defining all transform pipelines ===")

    # STEP 1
    LOGGER("P02", "step 1 — loading normalization stats...")

    mean, std = P02_F04_loadNormalizationStatsFromJsonFile(
        M02_P01_OUTPUT_NORM_STATS_JSON
    )

    # STEP 2
    LOGGER("P02", "step 2 — building train transform...")

    train_transform = P02_F05_buildTrainTransformWithAugmentation(
        mean,
        std,
        M02_P01_P02_INPUT_IMAGE_SIZE
    )

    # STEP 3
    LOGGER("P02", "step 3 — building val/test transform...")

    val_transform = P02_F06_buildValAndTestTransformNoAugmentation(
        mean,
        std,
        M02_P01_P02_INPUT_IMAGE_SIZE
    )

    # STEP 4
    LOGGER("P02", "step 4 — building inference transform...")

    inf_transform = P02_F07_buildInferenceTransformForSingleImage(
        mean,
        std,
        M02_P01_P02_INPUT_IMAGE_SIZE
    )

    LOGGER("P02", "=== M02_P02 done — all three transforms are ready ===")

    # sanity check
    LOGGER("P02", "sanity check — applying train transform to one real image...")

    # pick first species folder
    first_species = sorted(os.listdir(M02_P01_P02_INPUT_TRAIN_FOLDER))[0]

    # pick first image inside that species
    first_image = sorted(
        os.listdir(os.path.join(M02_P01_P02_INPUT_TRAIN_FOLDER, first_species))
    )[0]

    # construct full image path
    img_path = os.path.join(
        M02_P01_P02_INPUT_TRAIN_FOLDER,
        first_species,
        first_image
    )

    # load image
    img = Image.open(img_path).convert("RGB")

    # apply transform
    tensor = train_transform(img)

    # log results
    LOGGER("P02", f"input image: {img.size[0]}×{img.size[1]} px  →  tensor shape: {list(tensor.shape)}")
    LOGGER("P02", f"tensor min: {tensor.min():.4f}   tensor max: {tensor.max():.4f}")
    LOGGER("P02", "if shape is [3, 224, 224] and values are roughly in [-2, +2] — all correct")