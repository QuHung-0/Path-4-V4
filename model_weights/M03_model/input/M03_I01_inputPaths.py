# ========= FILE NAME: M03_I01_inputPaths.py =========
# FILE ROLE: Tells M03 where its inputs live — reads from M00, adds nothing new
# TOTAL FUNCTIONS IN THIS FILE: 0  (re-exports only)
# FUNCTION INDEX RANGE: none

# import the central path constants from M00
# these are the file and folder locations M03 will use for training, saving weights, and logging
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M03_P03_INPUT_TRAIN_FOLDER,     # training folder created by M01_P03
    M03_P03_INPUT_VAL_FOLDER,       # validation folder created by M01_P03
    M03_P03_INPUT_TEST_FOLDER,      # test folder created by M01_P03
    M03_P03_INPUT_NORM_STATS_JSON,  # normalization stats JSON created by M02_P01

    M03_P06_INPUT_NORM_STATS_JSON,  # same normalization JSON reused by k-fold workflow
    M03_P06_INPUT_BALANCED_FOLDER,   # balanced folder input for k-fold workflow (name contains a typo in the source)

    M03_P03_OUTPUT_CLASS_TO_IDX_JSON,  # output JSON mapping class name → numeric index

    M03_P05_OUTPUT_BEST_MODEL_PTH,     # output path for best model checkpoint
    M03_P05_OUTPUT_FINAL_MODEL_PTH,    # output path for final model checkpoint

    M03_P05_OUTPUT_TRAINING_LOG_CSV,   # output CSV for training log
    M03_P06_OUTPUT_KFOLD_LOG_CSV,      # output CSV for k-fold results

    M03_OUTPUT_WEIGHTS_FOLDER,         # folder where model weights are stored
    M03_OUTPUT_LOGS_FOLDER,            # folder where training logs are stored
)

# import the image transform functions built in M02
# M03 does not redefine preprocessing from scratch; it reuses the same transform recipes
from dataset.M02_preprocessing.process.M02_P02_defineAllTransformPipelines import (
    P02_F04_loadNormalizationStatsFromJsonFile,          # loads mean/std from JSON
    P02_F05_buildTrainTransformWithAugmentation,         # train augmentation pipeline
    P02_F06_buildValAndTestTransformNoAugmentation,      # validation/test pipeline
    P02_F07_buildInferenceTransformForSingleImage        # single-image inference pipeline
)

# import hyperparameters from the central settings file
# these control training behavior and model size
from config.M00_pipeline_config.M00_C02_allSettings import (
    RANDOM_SEED,      # global random seed for reproducibility

    M03_INPUT_IMAGE_SIZE,  # input size expected by the model

    M03_NUM_CLASSES,   # number of species classes
    M03_BATCH_SIZE,    # number of images per batch
    M03_LEARNING_RATE, # optimizer step size
    M03_NUM_EPOCHS,    # maximum number of training epochs
    M03_PATIENCE,      # early stopping patience
    M03_KFOLD_K,       # number of folds for k-fold validation
)

# import logging helpers so M03 can print progress and warnings in the same format as the rest of the project
from config.M00_pipeline_config.M00_C03_logger import (
    LG00_F01_printStepToConsole as LOGGER,      # normal progress logger
    LG00_F02_printWarningToConsole as WARN_LOGGER,  # warning logger
)