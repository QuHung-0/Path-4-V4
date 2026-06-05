# ========= FILE NAME: M07_P02_handleUploadImageAndReturnPrediction.py =========
# FILE ROLE: Receives uploaded image, validates it, predicts, stores result in server session
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P02_F04 → P02_F07

# import OS utilities for file path handling
import os

# import UUID for generating unique temporary filenames
import uuid

# import PIL image class for validation
from PIL import Image as PILImage

# import Flask current_app to access startup-loaded config
from flask import current_app

# import shared configuration and logger
from webapp.M07_web_app.input.M07_I01_appConfig import (
    M07_P02_OUTPUT_UPLOADS_FOLDER,  # folder where uploads are saved

    LOGGER,                         # logger
)

# import model inference helpers from M03
from model_weights.M03_model.process.M03_P07_loadModelAndPredictOneSingleImage import (
    P07_F24_preprocessOneSingleImageForInference,      # preprocess image for model
    P07_F25_runForwardPassAndGetSoftmaxProbabilities,  # run model and softmax
    P07_F26_convertSoftmaxOutputToLabelAndConfidenceScore,  # convert output to label/confidence
)

# import semantic enrichment helper from M05
from semantic.M05_semantic.process.M05_P03_enrichOnePredictionWithTaxonomyFromCache import (
    P03_F11_buildEnrichedPredictionDictionaryFromResult,  # build taxonomically enriched result
)

# import image hashing helper from M05
from semantic.M05_semantic.process.M05_P04_buildAndWriteJsonRecordForOneInference import (
    P04_F12_generateContentHashFromImageFileBytes,  # create SHA-256 image ID
)

# allowed file extensions for upload
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# FUNCTION 4
def P02_F04_receiveUploadedImageFileAndSaveToTempFolder(request) -> tuple:
    # reads the uploaded file from the POST request and saves it with a UUID filename
    # returns (saved_path, error_message) — error_message is None if all is ok

    # verify the request actually contains a file field
    if "file" not in request.files:
        return None, "No file in request"

    # get the uploaded file object
    file = request.files["file"]

    # reject empty file selection
    if file.filename == "":
        return None, "No file selected"

    # check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return None, f"File type not supported: {ext}"

    # make sure upload folder exists
    os.makedirs(M07_P02_OUTPUT_UPLOADS_FOLDER, exist_ok=True)

    # save the file using a random UUID name to avoid collisions
    save_name = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(M07_P02_OUTPUT_UPLOADS_FOLDER, save_name)
    file.save(save_path)

    LOGGER("P02", f"file saved to uploads: {save_name}")
    return save_path, None


# FUNCTION 5
def P02_F05_validateImageIsRgbAndMeetsMinimumSizeRequirement(image_path: str) -> tuple:
    # opens the image and checks it is RGB and at least 50×50 pixels
    # returns (is_valid, error_message)

    try:
        # open the image and convert to RGB so downstream processing is consistent
        img = PILImage.open(image_path).convert("RGB")
        w, h = img.size

        # reject tiny images because they may not be useful for classification
        if w < 50 or h < 50:
            return False, f"Image too small: {w}×{h} — minimum is 50×50"

        return True, None

    except Exception as e:
        # return a readable error if the file cannot be opened as an image
        return False, f"Cannot open image: {e}"


# FUNCTION 6
def P02_F06_runPretrainedModelOnImageAndGetLabelAndConfidence(
    image_path: str
) -> tuple:
    # uses the model already loaded in app.config — no reload per request

    # fetch startup-loaded objects from Flask config
    model = current_app.config["MODEL"]
    transform = current_app.config["TRANSFORM"]
    idx_to_class = current_app.config["IDX_TO_CLASS"]
    device = current_app.config["DEVICE"]

    # preprocess the uploaded image
    tensor = P07_F24_preprocessOneSingleImageForInference(image_path, transform)

    # run prediction and convert to probabilities
    probs = P07_F25_runForwardPassAndGetSoftmaxProbabilities(model, tensor, device)

    # decode predicted label and confidence score
    label, conf = P07_F26_convertSoftmaxOutputToLabelAndConfidenceScore(probs, idx_to_class)

    LOGGER("P02", f"predicted: {label}  confidence: {conf*100:.2f}%")
    return label, conf


# FUNCTION 7
def P02_F07_storePredictionResultInServerSideSessionNotForm(
    session, image_path: str, label: str, confidence: float
) -> dict:
    # builds enriched prediction and stores it in Flask server-side session
    # storing server-side means user cannot tamper with the data before saving to Neo4j

    # load taxonomy cache and threshold from app config
    taxonomy_cache = current_app.config["TAXONOMY_CACHE"]
    threshold = current_app.config["THRESHOLD"]

    # look up taxonomy for the predicted label
    taxon = taxonomy_cache.get(label)

    # build enriched semantic result
    enriched = P03_F11_buildEnrichedPredictionDictionaryFromResult(label, confidence, taxon)

    # generate a stable content hash for the uploaded image
    image_id = P04_F12_generateContentHashFromImageFileBytes(image_path)

    # assemble the final record that will later be saved to Neo4j
    result = {
        "image_id": image_id,
        "image_filename": os.path.basename(image_path),
        "internal_label": enriched["internal_label"],
        "confidence": enriched["confidence"],
        "taxonomy_status": enriched["taxonomy_status"],
        "scientific_name": enriched["scientific_name"],
        "authority": enriched.get("authority"),
        "aphia_id": enriched["aphia_id"],
        "lineage": enriched["lineage"],
        "model_version": "best_model.pth",
        "source_dataset": "upload",
        "threshold_used": threshold,
    }

    # store the result server-side so later requests can write it to Neo4j safely
    session["last_prediction"] = result
    LOGGER("P02", f"prediction stored in server session — status: {result['taxonomy_status']}")
    return result