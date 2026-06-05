# ========= FILE NAME: M03_P03_buildDataloadersForTrainingAndValidation.py =========
# FILE ROLE: Wraps the split folders in PyTorch DataLoaders and saves the class-to-index map
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P03_F08 → P03_F11

# import OS utilities for folder and file paths
import os

# import JSON module for saving and loading class mappings
import json

# import ImageFolder dataset helper from torchvision
from torchvision import datasets

# import DataLoader for batching data during training and evaluation
from torch.utils.data import DataLoader

# import project configuration, transforms, and logger
from model_weights.M03_model.input.M03_I01_inputPaths import (
    M03_P03_INPUT_TRAIN_FOLDER,   # training folder from M01_P03
    M03_P03_INPUT_VAL_FOLDER,     # validation folder from M01_P03
    M03_P03_INPUT_TEST_FOLDER,    # test folder from M01_P03
    M03_P03_INPUT_NORM_STATS_JSON,# normalization stats from M02_P01

    P02_F04_loadNormalizationStatsFromJsonFile,       # load normalization mean/std
    P02_F05_buildTrainTransformWithAugmentation,      # build train augmentation pipeline
    P02_F06_buildValAndTestTransformNoAugmentation,   # build val/test pipeline

    M03_P03_OUTPUT_CLASS_TO_IDX_JSON,  # output JSON for class mapping

    M03_INPUT_IMAGE_SIZE,  # image size expected by the model
    M03_BATCH_SIZE,        # batch size

    LOGGER                 # logging function
)


# FUNCTION 8
def P03_F08_createImageFolderDatasetFromPath(folder_path: str, transform) -> datasets.ImageFolder:
    # creates a PyTorch ImageFolder — automatically reads subfolders as class labels

    # ImageFolder expects folder_path/class_name/image.jpg structure
    dataset = datasets.ImageFolder(root=folder_path, transform=transform)

    # log how many images and classes were found
    LOGGER("P03", f"dataset loaded: {folder_path}  ({len(dataset)} images, {len(dataset.classes)} classes)")

    return dataset


# FUNCTION 9
def P03_F09_saveClassToIndexMappingToJsonFile(dataset: datasets.ImageFolder, output_path: str) -> None:
    # saves {class_name: index} so inference can convert model output numbers back to species names

    # make sure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # write the class_to_idx dictionary into JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset.class_to_idx, f, indent=4, sort_keys=True)

    # confirm the file was saved
    LOGGER("P03", f"class_to_idx saved → {output_path}")

    # print the mapping in index order so the log is easy to read
    for name, idx in sorted(dataset.class_to_idx.items(), key=lambda x: x[1]):
        LOGGER("P03", f"  {idx}: {name}")


# FUNCTION 10
def P03_F10_wrapDatasetInDataloaderWithBatchSizeAndShuffle(
    dataset, batch_size: int, shuffle: bool
) -> DataLoader:
    # wraps dataset in a DataLoader that feeds the model batch_size images at a time

    # create the loader with a single worker to keep behavior simple and reproducible
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    # log loader setup
    LOGGER("P03", f"dataloader ready — {len(dataset)} images, batch={batch_size}, shuffle={shuffle}")

    return loader


# FUNCTION 11
def P03_F11_loadClassToIndexMappingFromJsonFile(json_path: str) -> dict:
    # reads class_to_idx JSON and returns it as a dict — used by inference scripts

    # open the mapping file
    with open(json_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    # flip mapping so it becomes {index: class_name}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # log the size of the mapping
    LOGGER("P03", f"class map loaded — {len(class_to_idx)} classes")

    return idx_to_class


# FUNCTION 12
def buildAllThreeDataloaders() -> tuple:
    # convenience function — builds train, val, test dataloaders in one call

    # load normalization statistics from disk
    mean, std = P02_F04_loadNormalizationStatsFromJsonFile(M03_P03_INPUT_NORM_STATS_JSON)

    # build the training transform with augmentation
    train_transform = P02_F05_buildTrainTransformWithAugmentation(mean, std, M03_INPUT_IMAGE_SIZE)

    # build the evaluation transform with no augmentation
    eval_transform = P02_F06_buildValAndTestTransformNoAugmentation(mean, std, M03_INPUT_IMAGE_SIZE)

    # load the training dataset
    train_dataset = P03_F08_createImageFolderDatasetFromPath(M03_P03_INPUT_TRAIN_FOLDER, train_transform)

    # load the validation dataset
    val_dataset = P03_F08_createImageFolderDatasetFromPath(M03_P03_INPUT_VAL_FOLDER, eval_transform)

    # load the test dataset
    test_dataset = P03_F08_createImageFolderDatasetFromPath(M03_P03_INPUT_TEST_FOLDER, eval_transform)

    # save the class mapping using the training dataset's folder names
    P03_F09_saveClassToIndexMappingToJsonFile(train_dataset, M03_P03_OUTPUT_CLASS_TO_IDX_JSON)

    # wrap each dataset in a DataLoader
    train_loader = P03_F10_wrapDatasetInDataloaderWithBatchSizeAndShuffle(train_dataset, M03_BATCH_SIZE, shuffle=True)
    val_loader   = P03_F10_wrapDatasetInDataloaderWithBatchSizeAndShuffle(val_dataset,   M03_BATCH_SIZE, shuffle=False)
    test_loader  = P03_F10_wrapDatasetInDataloaderWithBatchSizeAndShuffle(test_dataset,  M03_BATCH_SIZE, shuffle=False)

    # return all three loaders together
    return train_loader, val_loader, test_loader


# MAIN
if __name__ == "__main__":
    # entry point when the script is run directly

    LOGGER("P03", "=== M03_P03 — building dataloaders ===")

    # build all loaders at once
    train_loader, val_loader, test_loader = buildAllThreeDataloaders()

    # print the number of batches in each loader
    LOGGER("P03", f"train batches: {len(train_loader)}")
    LOGGER("P03", f"val batches:   {len(val_loader)}")
    LOGGER("P03", f"test batches:  {len(test_loader)}")

    # inspect one batch from the training loader as a sanity check
    LOGGER("P03", "checking one batch from train loader...")
    images, labels = next(iter(train_loader))

    # log tensor shapes so you can verify the loader is working
    LOGGER("P03", f"batch image shape: {list(images.shape)}   labels shape: {list(labels.shape)}")
    LOGGER("P03", f"label values in this batch: {labels.tolist()}")

    LOGGER("P03", "=== M03_P03 done ===")