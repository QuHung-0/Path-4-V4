# ========= FILE NAME: M02_I01_inputPaths.py =========
# FILE ROLE: Tells M02 where its inputs and outputs live — reads from M00, adds nothing new
# TOTAL FUNCTIONS IN THIS FILE: 0
# FUNCTION INDEX RANGE: none

# import path constants from central pipeline config
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M02_P01_P02_INPUT_TRAIN_FOLDER,    # input: training dataset folder (from M01 split)
    M02_P01_OUTPUT_NORM_STATS_JSON     # output: JSON file storing normalization stats
)

# import preprocessing-related settings
from config.M00_pipeline_config.M00_C02_allSettings import (
    M02_P01_P02_INPUT_IMAGE_SIZE       # target image size (e.g., 224 for ResNet)
)

# import logging utilities
from config.M00_pipeline_config.M00_C03_logger import (
    LG00_F01_printStepToConsole as LOGGER,        # normal logging function
    LG00_F02_printWarningToConsole as WARN_LOGGER # warning logging function
)