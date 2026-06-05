# ========= FILE NAME: M07_I01_appConfig.py =========
# FILE ROLE: Flask application configuration — reads all settings from M00
# TOTAL FUNCTIONS IN THIS FILE: 0  (re-exports only)
# FUNCTION INDEX RANGE: none
# ALL THE FILE CONNECT TO (1): M00_pipeline_config/M00_C02_allSettings.py


# import file paths for model files, taxonomy cache, templates, static files, and uploads
from config.M00_pipeline_config.M00_C01_allPipelinePaths import (
    M07_P01_INPUT_BEST_MODEL_PTH,        # trained model weights used at app startup
    M07_P01_INPUT_CLASS_TO_IDX_JSON,     # class-to-index mapping used for label decoding
    M07_P01_INPUT_NORM_STATS_JSON,       # normalization statistics for preprocessing
    M07_P01_INPUT_TAXONOMY_MASTER_JSON,  # cached WoRMS taxonomy master JSON
    M07_P01_INPUT_THRESHOLD_JSON,        # confidence threshold chosen from M04

    M07_P01_OUTPUT_TEMPLATES_FOLDER,     # folder containing Flask HTML templates
    M07_P01_OUTPUT_STATIC_FOLDER,        # folder containing static CSS/JS/images

    M07_P02_OUTPUT_UPLOADS_FOLDER,       # folder where uploaded images are stored temporarily
)

# import runtime settings for image size, Neo4j, and Flask server configuration
from config.M00_pipeline_config.M00_C02_allSettings import (
    M02_P01_P02_INPUT_IMAGE_SIZE as M07_P01_INPUT_IMAGE_SIZE,  # reuse the model input size here

    M06_NEO4J_DATABASE as M07_P03_NEO4J_DATABASE,              # Neo4j database name for app queries

    M07_FLASK_HOST,              # host address where Flask listens
    M07_FLASK_PORT,              # port number where Flask listens
    M07_FLASK_DEBUG,             # debug mode on/off
    M07_FLASK_SECRET_KEY,        # secret key for Flask sessions
    M07_UPLOAD_MAX_AGE_MINS,     # how long uploads may remain before cleanup
)

# import shared logger functions for status and warning messages
from config.M00_pipeline_config.M00_C03_logger import (
    LG00_F01_printStepToConsole as LOGGER,      # normal progress logger
    LG00_F02_printWarningToConsole as WARN_LOGGER,  # warning logger
)