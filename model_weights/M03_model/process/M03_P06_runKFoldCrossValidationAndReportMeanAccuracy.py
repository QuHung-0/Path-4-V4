# ========= FILE NAME: M03_P06_runKFoldCrossValidationAndReportMeanAccuracy.py =========
# FILE ROLE: Trains the model 5 times on different splits and reports mean accuracy with std
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P06_F19 → P06_F22

# import OS module for directory creation
import os

# import CSV module for saving fold results
import csv

# import NumPy for mean and standard deviation calculations
import numpy as np

# import PyTorch loss and optimizer modules
import torch.nn as nn
import torch.optim as optim

# import data wrapper utilities for subsets and dataloaders
from torch.utils.data import DataLoader, Subset

# import torchvision ImageFolder for folder-based datasets
from torchvision import datasets

# import StratifiedKFold so each fold keeps class balance
from sklearn.model_selection import StratifiedKFold

# import shared config, transforms, and logger
from model_weights.M03_model.input.M03_I01_inputPaths import (
    P02_F04_loadNormalizationStatsFromJsonFile,     # load mean/std from JSON
    P02_F05_buildTrainTransformWithAugmentation,    # train transform with augmentation
    P02_F06_buildValAndTestTransformNoAugmentation, # eval transform without augmentation

    M03_P06_INPUT_NORM_STATS_JSON,                  # normalization JSON path
    M03_P06_INPUT_BALANCED_FOLDER,                   # balanced folder path (spelling typo kept from source)

    M03_P06_OUTPUT_KFOLD_LOG_CSV,                   # CSV output for k-fold results

    RANDOM_SEED,                                    # seed value

    M03_INPUT_IMAGE_SIZE,                           # image size

    M03_PATIENCE,                                   # early stopping patience
    M03_NUM_CLASSES,                                # number of output classes
    M03_BATCH_SIZE,                                 # batch size
    M03_LEARNING_RATE,                              # learning rate
    M03_NUM_EPOCHS,                                 # max epochs
    M03_KFOLD_K,                                    # number of folds

    LOGGER,                                         # logger
)

# import device/seed setup helpers
from M03_P01_checkGpuAndSetReproducibilitySeed import (
    P01_F01_checkIfGpuIsAvailableAndPrintItsName,   # device detection
    P01_F02_setAllRandomSeedsForFullReproducibility,# seed setup
)

# import model helpers
from M03_P02_buildResNet18WithCustomOutputLayer import (
    P02_F04_loadPretrainedResNet18FromPytorchHub,       # pretrained backbone
    P02_F05_replaceLastFullyConnectedLayerWithNOutputs, # classifier replacement
)

# import training/validation epoch helpers
from M03_P04_trainOneEpochAndReturnLossAndAccuracy import (
    P04_F12_runOneTrainingEpochAndUpdateWeights,    # training pass
    P04_F13_runOneValidationEpochWithNoGradients,   # validation pass
)


# FUNCTION 19
def P06_F19_loadBalancedDatasetWithEvalTransform(balanced_folder: str):
    # loads the full balanced_200 dataset (2000 images) with no augmentation for k-fold

    # load normalization stats
    mean, std = P02_F04_loadNormalizationStatsFromJsonFile(M03_P06_INPUT_NORM_STATS_JSON)

    # build deterministic evaluation transform
    transform = P02_F06_buildValAndTestTransformNoAugmentation(mean, std, M03_INPUT_IMAGE_SIZE)

    # create dataset from folder structure
    dataset = datasets.ImageFolder(root=balanced_folder, transform=transform)

    # log dataset size and class count
    LOGGER("P06", f"k-fold dataset: {len(dataset)} images, {len(dataset.classes)} classes")

    return dataset


# FUNCTION 20
def P06_F20_trainAndEvaluateModelOnOneFold(
    dataset, train_indices, val_indices, device, fold_num: int
) -> float:
    # trains a fresh model on one fold's training indices, evaluates on val indices
    # returns the best validation accuracy achieved during training this fold

    # load normalization stats again so we can build the train transform
    mean, std = P02_F04_loadNormalizationStatsFromJsonFile(M03_P06_INPUT_NORM_STATS_JSON)

    # build augmentation transform for fold training subset
    train_transform = P02_F05_buildTrainTransformWithAugmentation(mean, std, M03_INPUT_IMAGE_SIZE)

    # recreate the dataset with augmentation enabled for training
    train_dataset_aug = datasets.ImageFolder(root=M03_P06_INPUT_BALANCED_FOLDER, transform=train_transform)

    # wrap selected training indices into a subset
    train_subset = Subset(train_dataset_aug, train_indices)

    # wrap selected validation indices into a subset using the non-augmented dataset
    val_subset = Subset(dataset, val_indices)

    # create dataloaders for this fold
    train_loader = DataLoader(train_subset, batch_size=M03_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=M03_BATCH_SIZE, shuffle=False, num_workers=0)

    # create a fresh pretrained ResNet-18 for this fold
    model = P02_F04_loadPretrainedResNet18FromPytorchHub()
    model = P02_F05_replaceLastFullyConnectedLayerWithNOutputs(model, M03_NUM_CLASSES)
    model = model.to(device)

    # define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=M03_LEARNING_RATE)

    # initialize fold tracking values
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    patience = M03_PATIENCE

    # train for up to the configured number of epochs
    for epoch in range(1, M03_NUM_EPOCHS + 1):

        # one training epoch
        P04_F12_runOneTrainingEpochAndUpdateWeights(model, train_loader, optimizer, loss_fn, device)

        # one validation epoch
        val_loss, val_acc = P04_F13_runOneValidationEpochWithNoGradients(model, val_loader, loss_fn, device)

        # log fold progress
        LOGGER("P06", f"  fold {fold_num} epoch {epoch}: val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")

        # track highest validation accuracy seen during training
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # reset patience if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

            # stop early if patience is exhausted
            if patience_counter >= patience:
                LOGGER("P06", f"  fold {fold_num} early stopping at epoch {epoch}")
                break

    # log the best accuracy found in this fold
    LOGGER("P06", f"  fold {fold_num} best val accuracy: {best_val_acc:.2f}%")

    # return best validation accuracy for this fold
    return best_val_acc


# FUNCTION 21
def P06_F21_runAllFoldsAndCollectAccuracyForEachFold(
    dataset, device, k: int, seed: int
) -> list:
    # runs k-fold CV and returns a list of k accuracy values (one per fold)

    # collect labels for stratified splitting
    labels = [label for _, label in dataset.samples]

    # define stratified k-fold splitter with shuffling and fixed seed
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    # list of all dataset indices
    indices = list(range(len(dataset)))

    # list of best accuracies from each fold
    fold_accuracies = []

    # iterate through each fold split
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(indices, labels), start=1):

        # print fold heading
        LOGGER("P06", f"─── k-fold: fold {fold_num}/{k} ─────────────────────────")

        # print split sizes
        LOGGER("P06", f"  train size: {len(train_idx)}   val size: {len(val_idx)}")

        # train and evaluate this fold
        acc = P06_F20_trainAndEvaluateModelOnOneFold(
            dataset, train_idx, val_idx, device, fold_num
        )

        # store fold result
        fold_accuracies.append(acc)

    # return list of fold accuracies
    return fold_accuracies


# FUNCTION 22
def P06_F22_computeMeanAndStdAcrossAllFoldsAndSaveReport(fold_accuracies: list) -> None:
    # computes mean ± std from k-fold results and saves to CSV

    # calculate mean accuracy across folds
    mean_acc = float(np.mean(fold_accuracies))

    # calculate standard deviation across folds
    std_acc = float(np.std(fold_accuracies))

    # print summary header
    LOGGER("P06", "─── k-fold results ─────────────────────────────────")

    # print each fold score
    for i, acc in enumerate(fold_accuracies, start=1):
        LOGGER("P06", f"  fold {i}: {acc:.2f}%")

    # print aggregate statistics
    LOGGER("P06", f"  MEAN: {mean_acc:.2f}%  ±  STD: {std_acc:.2f}%")
    LOGGER("P06", "────────────────────────────────────────────────────")

    # ensure output folder exists
    os.makedirs(os.path.dirname(M03_P06_OUTPUT_KFOLD_LOG_CSV), exist_ok=True)

    # save results to CSV
    with open(M03_P06_OUTPUT_KFOLD_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "val_accuracy_percent"])
        for i, acc in enumerate(fold_accuracies, start=1):
            writer.writerow([i, f"{acc:.2f}"])
        writer.writerow(["MEAN", f"{mean_acc:.2f}"])
        writer.writerow(["STD", f"{std_acc:.2f}"])

    # confirm output location
    LOGGER("P06", f"k-fold report saved → {M03_P06_OUTPUT_KFOLD_LOG_CSV}")


# MAIN
if __name__ == "__main__":
    # entry point for running k-fold evaluation directly

    LOGGER("P06", "=== M03_P06 — k-fold cross validation ===")
    LOGGER("P06", f"k={M03_KFOLD_K} folds, seed={RANDOM_SEED}")
    LOGGER("P06", "NOTE: this trains the model k times — expect ~2-3 hours total on GTX 1650")

    # select device
    device = P01_F01_checkIfGpuIsAvailableAndPrintItsName()

    # set seeds
    P01_F02_setAllRandomSeedsForFullReproducibility(RANDOM_SEED)

    # load dataset once for fold splitting
    dataset = P06_F19_loadBalancedDatasetWithEvalTransform(M03_P06_INPUT_BALANCED_FOLDER)

    # run all folds and collect accuracies
    fold_accuracies = P06_F21_runAllFoldsAndCollectAccuracyForEachFold(dataset, device, M03_KFOLD_K, RANDOM_SEED)

    # compute and save fold statistics
    P06_F22_computeMeanAndStdAcrossAllFoldsAndSaveReport(fold_accuracies)

    LOGGER("P06", "=== M03_P06 done ===")