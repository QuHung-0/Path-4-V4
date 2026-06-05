# ========= FILE NAME: M03_P01_checkGpuAndSetReproducibilitySeed.py =========
# FILE ROLE: Confirms GPU is available and locks all random sources to the same seed
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P01_F01 → P01_F03

# import Python's random module for seed control
import random

# import NumPy for seed control and numerical consistency
import numpy as np

# import PyTorch for GPU detection and deep learning operations
import torch

# import project-wide paths, settings, and logger from M03 input config
from model_weights.M03_model.input.M03_I01_inputPaths import (
    RANDOM_SEED,        # global seed value used throughout the project

    M03_INPUT_IMAGE_SIZE,  # image size expected by the model

    M03_NUM_CLASSES,   # number of output classes
    M03_BATCH_SIZE,    # batch size
    M03_LEARNING_RATE, # learning rate
    M03_NUM_EPOCHS,    # training epochs
    M03_PATIENCE,      # early stopping patience
    M03_KFOLD_K,       # k-fold count

    LOGGER             # logging function
)


# FUNCTION 1
def P01_F01_checkIfGpuIsAvailableAndPrintItsName() -> torch.device:
    # checks for CUDA GPU and returns the device to use for all tensor operations

    if torch.cuda.is_available():
        # if a GPU exists, use it
        device = torch.device("cuda")

        # get the GPU name, for example "NVIDIA GeForce RTX ..."
        gpu_name = torch.cuda.get_device_name(0)

        # get total GPU memory in gigabytes
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        # print a human-readable GPU summary
        LOGGER("P01", f"GPU found: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    else:
        # if no GPU exists, fall back to CPU
        device = torch.device("cpu")

        # warn the user that training will be slower
        LOGGER("P01", "WARNING — no GPU found, training will run on CPU (much slower)")

    # return the chosen device so the rest of training can use it
    return device


# FUNCTION 2
def P01_F02_setAllRandomSeedsForFullReproducibility(seed: int) -> None:
    # sets the same seed in Python, NumPy, and PyTorch so every run gives identical results

    # seed Python's built-in random module
    random.seed(seed)

    # seed NumPy's random generator
    np.random.seed(seed)

    # seed PyTorch CPU operations
    torch.manual_seed(seed)

    # seed PyTorch CUDA operations across all GPUs
    torch.cuda.manual_seed_all(seed)

    # force cuDNN to use deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True

    # disable cuDNN autotuning because it can introduce run-to-run variation
    torch.backends.cudnn.benchmark = False

    # log the seed choice
    LOGGER("P01", f"all random seeds set to {seed} — results will be reproducible")


# FUNCTION 3
def P01_F03_printAllHyperparametersBeforeTrainingStarts() -> None:
    # prints every setting so the terminal log is a full record of what this run used

    LOGGER("P01", "─── hyperparameters ───────────────────────────────")
    LOGGER("P01", f"  NUM_CLASSES:    {M03_NUM_CLASSES}")
    LOGGER("P01", f"  INPUT_SIZE:     {M03_INPUT_IMAGE_SIZE}×{M03_INPUT_IMAGE_SIZE}")
    LOGGER("P01", f"  BATCH_SIZE:     {M03_BATCH_SIZE}")
    LOGGER("P01", f"  LEARNING_RATE:  {M03_LEARNING_RATE}")
    LOGGER("P01", f"  NUM_EPOCHS:     {M03_NUM_EPOCHS}")
    LOGGER("P01", f"  PATIENCE:       {M03_PATIENCE}")
    LOGGER("P01", f"  KFOLD_K:        {M03_KFOLD_K}")
    LOGGER("P01", f"  RANDOM_SEED:    {RANDOM_SEED}")
    LOGGER("P01", "───────────────────────────────────────────────────")


# MAIN
if __name__ == "__main__":
    # entry point when this file is run directly

    LOGGER("P01", "=== M03_P01 — GPU check and seed setup ===")

    # detect the best compute device
    device = P01_F01_checkIfGpuIsAvailableAndPrintItsName()

    # lock randomness for reproducible training
    P01_F02_setAllRandomSeedsForFullReproducibility(RANDOM_SEED)

    # print the training settings into the log
    P01_F03_printAllHyperparametersBeforeTrainingStarts()

    # confirm the chosen device
    LOGGER("P01", f"device ready: {device}")

    LOGGER("P01", "=== M03_P01 done ===")