# ========= FILE NAME: M01_I01_inputPaths.py =========
# FILE ROLE: Tells M01 where its raw input data lives — reads from M00, adds nothing new
# TOTAL FUNCTIONS IN THIS FILE: 0
# FUNCTION INDEX RANGE: none


# import path constants from the central pipeline config file (M00)
# these variables define where data is stored and where outputs should go
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M01_INPUT_RAW_FISH_IMAGE_FOLDER,      # path to the original raw dataset of fish images

    M01_P02_INPUT_SPECIES_MAP,            # path to JSON file mapping folder IDs → species names

    M01_P02_OUTPUT_BALANCED_FOLDER,      # output folder where balanced dataset (200 images/species) will be saved

    M01_P03_OUTPUT_TRAIN_FOLDER,         # output folder for training split
    M01_P03_OUTPUT_VAL_FOLDER,           # output folder for validation split
    M01_P03_OUTPUT_TEST_FOLDER,          # output folder for test split

    M01_OUTPUT_REPORTS_FOLDER,           # folder where reports (CSV, markdown, etc.) are stored

    M01_P01_OUTPUT_EXPLORATION_CSV,      # CSV file path for dataset exploration results

    M01_P04_OUTPUT_DISTRIBUTION_PNG,     # PNG file path for class distribution chart
    M01_P04_OUTPUT_SUMMARY_MD,           # Markdown file path for dataset summary


    M01_P05_OUTPUT_REMAINING_FOLDER,
    M01_P05_OUTPUT_REMAINING_REPORT_TXT
)

# import numeric settings from central settings file (M00)
# these define rules for dataset filtering and splitting
from config.M00_pipeline_config.M00_C02_allSettings import (
    M01_P02_MIN_IMAGES_PER_SPECIES,      # minimum number of images required to keep a species

    M01_P02_SAMPLE_SIZE_PER_SPECIES,     # number of images to sample per species after balancing

    M01_P03_TRAIN_COUNT,                # number of images per species in training set
    M01_P03_VAL_COUNT,                  # number of images per species in validation set
    M01_P03_TEST_COUNT,                 # number of images per species in test set

    RANDOM_SEED,                        # fixed seed to make randomness reproducible
)

# import logging functions from logger module (M00)
# rename them for easier use inside M01
from config.M00_pipeline_config.M00_C03_logger import (
    LG00_F01_printStepToConsole as LOGGER,       # function to print normal progress messages
    LG00_F02_printWarningToConsole as WARN_LOGGER,  # function to print warning messages
)