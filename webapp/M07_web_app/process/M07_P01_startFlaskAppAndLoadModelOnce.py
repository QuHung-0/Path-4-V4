# ========= FILE NAME: M07_P01_startFlaskAppAndLoadModelOnce.py =========
# FILE ROLE: Creates the Flask app and loads model, transforms, class map, threshold at startup
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P01_F01 → P01_F03

# import OS utilities for creating folders
import os

# import JSON for loading taxonomy cache and threshold
import json

# import PyTorch for model loading and device selection
import torch

# import Flask application object
from flask import Flask

# import all app-level configuration values and logger
from webapp.M07_web_app.input.M07_I01_appConfig import (
    M07_P01_INPUT_BEST_MODEL_PTH,        # best trained model weights
    M07_P01_INPUT_CLASS_TO_IDX_JSON,     # class mapping JSON
    M07_P01_INPUT_NORM_STATS_JSON,       # normalization stats JSON
    M07_P01_INPUT_TAXONOMY_MASTER_JSON,  # cached taxonomy master JSON
    M07_P01_INPUT_THRESHOLD_JSON,        # threshold JSON from M04

    M07_P01_OUTPUT_TEMPLATES_FOLDER,     # templates folder for Flask
    M07_P01_OUTPUT_STATIC_FOLDER,        # static folder for Flask

    M07_P02_OUTPUT_UPLOADS_FOLDER,      # upload folder for user images

    M07_P01_INPUT_IMAGE_SIZE,           # model input image size
    M07_FLASK_SECRET_KEY,               # Flask secret key

    LOGGER                              # logger
)

# import model loading helper from M03
from model_weights.M03_model.process.M03_P07_loadModelAndPredictOneSingleImage import (
    P07_F23_loadTrainedModelWeightsFromFile,  # loads trained ResNet-18 weights
)

# import class map loader from M03
from model_weights.M03_model.process.M03_P03_buildDataloadersForTrainingAndValidation import (
    P03_F11_loadClassToIndexMappingFromJsonFile,  # loads idx→class map
)

# import normalization and inference transform helpers from M02
from dataset.M02_preprocessing.process.M02_P02_defineAllTransformPipelines import (
    P02_F04_loadNormalizationStatsFromJsonFile,   # loads mean/std
    P02_F07_buildInferenceTransformForSingleImage, # builds single-image inference transform
)


# FUNCTION 1
def P01_F01_loadTrainedModelIntoMemoryAtAppStartup(app: Flask) -> None:
    # loads model weights once when Flask starts — never loads again per request
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # choose GPU if available
    model = P07_F23_loadTrainedModelWeightsFromFile(M07_P01_INPUT_BEST_MODEL_PTH, device)  # load model
    app.config["MODEL"] = model   # store model in Flask app config
    app.config["DEVICE"] = device # store device in Flask app config
    LOGGER("P01", f"model loaded at startup on {device}")


# FUNCTION 2
def P01_F02_loadNormStatsAndClassMapAtAppStartup(app: Flask) -> None:
    # loads norm stats and class index map once at startup
    mean, std = P02_F04_loadNormalizationStatsFromJsonFile(M07_P01_INPUT_NORM_STATS_JSON)  # load mean/std
    transform = P02_F07_buildInferenceTransformForSingleImage(mean, std, M07_P01_INPUT_IMAGE_SIZE)  # build transform
    idx_to_class = P03_F11_loadClassToIndexMappingFromJsonFile(M07_P01_INPUT_CLASS_TO_IDX_JSON)      # load class map

    # load the cached taxonomy dictionary from disk
    with open(M07_P01_INPUT_TAXONOMY_MASTER_JSON, "r", encoding="utf-8") as f:
        taxonomy_cache = json.load(f)

    # save all loaded objects inside Flask config for later reuse
    app.config["TRANSFORM"] = transform
    app.config["IDX_TO_CLASS"] = idx_to_class
    app.config["TAXONOMY_CACHE"] = taxonomy_cache
    LOGGER("P01", "transforms, class map, taxonomy cache loaded at startup")


# FUNCTION 3
def P01_F03_loadConfidenceThresholdAtAppStartup(app: Flask) -> None:
    # reads the data-derived threshold from M04 output — never hardcoded
    with open(M07_P01_INPUT_THRESHOLD_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    app.config["THRESHOLD"] = data["threshold"]  # store threshold in Flask config
    LOGGER("P01", f"confidence threshold loaded: {data['threshold']} (from M04 PR curve)")


# FUNCTION 4
def createFlaskApp() -> Flask:
    # builds and configures the Flask application object

    # create required folders before the app starts handling requests
    os.makedirs(M07_P02_OUTPUT_UPLOADS_FOLDER, exist_ok=True)
    os.makedirs(M07_P01_OUTPUT_TEMPLATES_FOLDER, exist_ok=True)
    os.makedirs(M07_P01_OUTPUT_STATIC_FOLDER, exist_ok=True)

    # create the Flask app and tell it where templates and static files are located
    app = Flask(
        __name__,
        template_folder=M07_P01_OUTPUT_TEMPLATES_FOLDER,
        static_folder=M07_P01_OUTPUT_STATIC_FOLDER,
    )

    # set Flask secret key for session management
    app.secret_key = M07_FLASK_SECRET_KEY

    # load all expensive resources once during startup
    LOGGER("P01", "loading model and config at startup...")
    P01_F01_loadTrainedModelIntoMemoryAtAppStartup(app)
    P01_F02_loadNormStatsAndClassMapAtAppStartup(app)
    P01_F03_loadConfidenceThresholdAtAppStartup(app)
    LOGGER("P01", "Flask app ready")

    return app