# ========= FILE NAME: M03_P05_trainFinalModelWithEarlyStoppingAndLogging.py =========
# FILE ROLE: Runs the full training loop, saves best model, logs every epoch to CSV
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P05_F15 → P05_F18

# import OS module for folder creation
import os

# import CSV module for writing training logs
import csv

# import PyTorch core
import torch

# import PyTorch neural-network utilities
import torch.nn as nn

# import PyTorch optimizer utilities
import torch.optim as optim

# import shared project paths, settings, and logger
from model_weights.M03_model.input.M03_I01_inputPaths import (
    M03_P05_OUTPUT_BEST_MODEL_PTH,      # where best checkpoint is saved
    M03_P05_OUTPUT_FINAL_MODEL_PTH,     # where final checkpoint is saved

    M03_P05_OUTPUT_TRAINING_LOG_CSV,    # where epoch-by-epoch log is saved

    M03_OUTPUT_WEIGHTS_FOLDER,          # weights directory

    RANDOM_SEED,                        # global seed

    M03_NUM_CLASSES,                    # number of output classes
    M03_LEARNING_RATE,                  # optimizer learning rate
    M03_NUM_EPOCHS,                     # maximum number of epochs
    M03_PATIENCE,                       # early stopping patience

    LOGGER,                             # normal logger
    WARN_LOGGER                         # warning logger
)

# import training setup helpers from M03_P01
from M03_P01_checkGpuAndSetReproducibilitySeed import (
    P01_F01_checkIfGpuIsAvailableAndPrintItsName,         # device selection
    P01_F02_setAllRandomSeedsForFullReproducibility,      # seed setting
    P01_F03_printAllHyperparametersBeforeTrainingStarts,  # log all hyperparameters
)

# import model-building helpers from M03_P02
from M03_P02_buildResNet18WithCustomOutputLayer import (
    P02_F04_loadPretrainedResNet18FromPytorchHub,            # load pretrained backbone
    P02_F05_replaceLastFullyConnectedLayerWithNOutputs,      # replace classifier head
)

# import dataloader builder from M03_P03
from M03_P03_buildDataloadersForTrainingAndValidation import buildAllThreeDataloaders

# import epoch-level training and validation helpers from M03_P04
from M03_P04_trainOneEpochAndReturnLossAndAccuracy import (
    P04_F12_runOneTrainingEpochAndUpdateWeights,     # one training epoch
    P04_F13_runOneValidationEpochWithNoGradients,    # one validation epoch
    P04_F14_saveBestModelWeightsWhenValLossImproves, # save best checkpoint
)


# FUNCTION 15
def P05_F15_initialiseTrainingLogCSVWithHeaders(log_path: str) -> None:
    # creates the CSV file and writes the header row before training starts

    # make sure the parent folder exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # open the CSV file in write mode so previous logs are overwritten
    with open(log_path, "w", newline="", encoding="utf-8") as f:

        # create a CSV writer
        writer = csv.writer(f)

        # write the header row for later plotting and analysis
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "saved_best"])


# FUNCTION 16
def P05_F16_appendOneEpochResultRowToTrainingLogCSV(
    log_path: str, epoch: int,
    train_loss: float, train_acc: float,
    val_loss: float,   val_acc: float,
    saved_best: bool
) -> None:
    # appends one row per epoch so you can plot the training curve afterwards

    # open CSV in append mode so new rows are added after the header
    with open(log_path, "a", newline="", encoding="utf-8") as f:

        # create CSV writer
        writer = csv.writer(f)

        # write one epoch summary row
        writer.writerow([
            epoch,
            f"{train_loss:.6f}", f"{train_acc:.2f}",
            f"{val_loss:.6f}",   f"{val_acc:.2f}",
            "yes" if saved_best else "no"
        ])


# FUNCTION 17
def P05_F17_checkEarlyStoppingConditionAndReturnNewCounter(
    val_loss: float, best_loss: float, counter: int, patience: int
) -> tuple:
    # increments patience counter if val loss did not improve, resets to 0 if it did
    # returns (new_counter, should_stop)

    # if validation loss improved, reset patience counter immediately
    if val_loss < best_loss:
        return 0, False

    # otherwise increase the no-improvement counter
    new_counter = counter + 1

    # if the counter reaches the patience limit, request early stop
    if new_counter >= patience:
        WARN_LOGGER("P05", f"early stopping triggered — no improvement for {patience} epochs")
        return new_counter, True

    # otherwise keep training
    WARN_LOGGER("P05", f"no improvement — patience counter: {new_counter}/{patience}")
    return new_counter, False


# FUNCTION 18
def P05_F18_runFullTrainingLoopForAllEpochsAndSaveBestModel(
    model, train_loader, val_loader, optimizer, loss_fn, device
) -> None:
    # the main training loop — runs up to NUM_EPOCHS rounds, stops early if val loss plateaus

    # initialize the best validation loss to a very large value
    best_loss = float("inf")

    # initialize patience counter
    patience_counter = 0

    # create the CSV log file and write header row
    P05_F15_initialiseTrainingLogCSVWithHeaders(M03_P05_OUTPUT_TRAINING_LOG_CSV)

    # loop through each epoch
    for epoch in range(1, M03_NUM_EPOCHS + 1):

        # print an epoch separator
        LOGGER("P05", f"─── epoch {epoch}/{M03_NUM_EPOCHS} ───────────────────────────")

        # run one training epoch
        train_loss, train_acc = P04_F12_runOneTrainingEpochAndUpdateWeights(
            model, train_loader, optimizer, loss_fn, device
        )

        # run one validation epoch
        val_loss, val_acc = P04_F13_runOneValidationEpochWithNoGradients(
            model, val_loader, loss_fn, device
        )

        # log the epoch results
        LOGGER("P05", f"  train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
        LOGGER("P05", f"  val    loss={val_loss:.4f}  acc={val_acc:.2f}%")

        # check improvement BEFORE updating best_loss
        improved = val_loss < best_loss

        # save the model if validation loss is the best so far
        best_loss = P04_F14_saveBestModelWeightsWhenValLossImproves(
            model, val_loss, best_loss, M03_P05_OUTPUT_BEST_MODEL_PTH
        )

        # append this epoch's results to CSV
        P05_F16_appendOneEpochResultRowToTrainingLogCSV(
            M03_P05_OUTPUT_TRAINING_LOG_CSV, epoch,
            train_loss, train_acc,
            val_loss, val_acc,
            saved_best=improved
        )

        # if validation improved, reset patience counter
        if improved:
            patience_counter = 0
            LOGGER("P05", f"  patience counter reset — new best found")
        else:
            # otherwise increment patience counter
            patience_counter += 1

            # stop if patience limit reached
            if patience_counter >= M03_PATIENCE:
                WARN_LOGGER("P05", f"early stopping triggered — no improvement for {M03_PATIENCE} epochs")
                break

            # log current patience status
            WARN_LOGGER("P05", f"no improvement — patience counter: {patience_counter}/{M03_PATIENCE}")

    # ensure output folder exists before saving final model
    os.makedirs(M03_OUTPUT_WEIGHTS_FOLDER, exist_ok=True)

    # save the last model state as the final model
    torch.save(model.state_dict(), M03_P05_OUTPUT_FINAL_MODEL_PTH)

    # log saved artifacts
    LOGGER("P05", f"final model weights saved → {M03_P05_OUTPUT_FINAL_MODEL_PTH}")
    LOGGER("P05", f"best model weights saved  → {M03_P05_OUTPUT_BEST_MODEL_PTH}")
    LOGGER("P05", f"training log saved        → {M03_P05_OUTPUT_TRAINING_LOG_CSV}")


# MAIN
if __name__ == "__main__":
    # entry point for running full training directly

    LOGGER("P05", "=== M03_P05 — full training run ===")

    # detect GPU or CPU
    device = P01_F01_checkIfGpuIsAvailableAndPrintItsName()

    # set random seeds
    P01_F02_setAllRandomSeedsForFullReproducibility(RANDOM_SEED)

    # print all hyperparameters so the run is fully documented
    P01_F03_printAllHyperparametersBeforeTrainingStarts()

    # build train / val / test dataloaders
    LOGGER("P05", "building dataloaders...")
    train_loader, val_loader, _ = buildAllThreeDataloaders()

    # build model
    LOGGER("P05", "building model...")
    model = P02_F04_loadPretrainedResNet18FromPytorchHub()
    model = P02_F05_replaceLastFullyConnectedLayerWithNOutputs(model, M03_NUM_CLASSES)
    model = model.to(device)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=M03_LEARNING_RATE)

    # start the training loop
    LOGGER("P05", "starting training — this will take ~30 minutes on GTX 1650...")
    P05_F18_runFullTrainingLoopForAllEpochsAndSaveBestModel(
        model, train_loader, val_loader, optimizer, loss_fn, device
    )

    LOGGER("P05", "=== M03_P05 done ===")