# ========= FILE NAME: M03_P07_loadModelAndPredictOneSingleImage.py =========
# FILE ROLE: Loads saved model weights and returns a species label + confidence for one image
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P07_F23 → P07_F26

# import OS module for file paths
import os

# import PyTorch core
import torch

# import PIL image loader
from PIL import Image

# import shared config and preprocessing helpers
from model_weights.M03_model.input.M03_I01_inputPaths import (
    P02_F04_loadNormalizationStatsFromJsonFile,   # load mean/std
    P02_F07_buildInferenceTransformForSingleImage, # inference-only transform

    M03_P03_INPUT_NORM_STATS_JSON,                # normalization JSON
    M03_P03_INPUT_TEST_FOLDER,                    # test folder (used in demo block)

    M03_P03_OUTPUT_CLASS_TO_IDX_JSON,             # class-to-index map
    M03_P05_OUTPUT_BEST_MODEL_PTH,                # saved best model weights

    M03_NUM_CLASSES,                              # number of output classes
    M03_INPUT_IMAGE_SIZE,                         # model input size

    LOGGER,                                       # logger
)

# import model-building helpers
from model_weights.M03_model.process.M03_P02_buildResNet18WithCustomOutputLayer import (
    P02_F04_loadPretrainedResNet18FromPytorchHub,       # backbone loader
    P02_F05_replaceLastFullyConnectedLayerWithNOutputs, # classifier replacement
)

# import class-map loader
from model_weights.M03_model.process.M03_P03_buildDataloadersForTrainingAndValidation import (
    P03_F11_loadClassToIndexMappingFromJsonFile,  # returns idx -> class mapping
)


# FUNCTION 23
def P07_F23_loadTrainedModelWeightsFromFile(weights_path: str, device: torch.device):
    # builds the ResNet-18 structure then loads the saved weights into it

    # create the same model architecture used during training
    model = P02_F04_loadPretrainedResNet18FromPytorchHub()
    model = P02_F05_replaceLastFullyConnectedLayerWithNOutputs(model, M03_NUM_CLASSES)

    # load trained parameter values from disk
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # move model to device
    model = model.to(device)

    # switch to evaluation mode
    model.eval()

    # log load success
    LOGGER("P07", f"model weights loaded from: {weights_path}")

    return model


# FUNCTION 24
def P07_F24_preprocessOneSingleImageForInference(
    image_path: str, transform
) -> torch.Tensor:
    # opens the image, converts to RGB, applies transform, adds batch dimension

    # load image from disk and force RGB mode
    img = Image.open(image_path).convert("RGB")

    # apply the preprocessing transform
    tensor = transform(img)

    # add batch dimension so shape becomes [1, C, H, W]
    tensor = tensor.unsqueeze(0)   # shape: [1, 3, 224, 224]

    return tensor


# FUNCTION 25
def P07_F25_runForwardPassAndGetSoftmaxProbabilities(
    model, tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
    # runs the image tensor through the model and converts raw scores to probabilities

    # move input tensor to the same device as the model
    tensor = tensor.to(device)

    # inference only, so gradients are disabled
    with torch.no_grad():

        # raw model output logits
        raw_output = model(tensor)  # shape: [1, 10]

        # convert logits into normalized probabilities
        probabilities = torch.softmax(raw_output, dim=1)  # values sum to 1.0

    return probabilities


# FUNCTION 26
def P07_F26_convertSoftmaxOutputToLabelAndConfidenceScore(
    probabilities: torch.Tensor, idx_to_class: dict
) -> tuple:
    # finds the highest probability, maps its index to a species name, returns (label, confidence)

    # get the max probability value and the index where it occurs
    confidence, predicted_idx = probabilities.max(dim=1)

    # map numeric prediction index back to species label
    label = idx_to_class[predicted_idx.item()]

    # convert tensor scalar to Python float
    conf_value = confidence.item()

    return label, conf_value


# convenience wrapper — loads model once and predicts one image end to end
def predictOneImage(image_path: str) -> tuple:
    # choose GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load normalization stats
    mean, std = P02_F04_loadNormalizationStatsFromJsonFile(M03_P03_INPUT_NORM_STATS_JSON)

    # build inference preprocessing pipeline
    transform = P02_F07_buildInferenceTransformForSingleImage(mean, std, M03_INPUT_IMAGE_SIZE)

    # load class mapping from index back to class name
    idx_to_class = P03_F11_loadClassToIndexMappingFromJsonFile(M03_P03_OUTPUT_CLASS_TO_IDX_JSON)

    # load trained model weights
    model = P07_F23_loadTrainedModelWeightsFromFile(M03_P05_OUTPUT_BEST_MODEL_PTH, device)

    # preprocess the image into a tensor
    tensor = P07_F24_preprocessOneSingleImageForInference(image_path, transform)

    # run model and get probabilities
    probs = P07_F25_runForwardPassAndGetSoftmaxProbabilities(model, tensor, device)

    # convert probabilities into label + confidence
    label, conf = P07_F26_convertSoftmaxOutputToLabelAndConfidenceScore(probs, idx_to_class)

    return label, conf


# MAIN
if __name__ == "__main__":
    # import sys even though it is not actually used here
    import sys

    LOGGER("P07", "=== M03_P07 — single image prediction test ===")

    # use the first test image found as a quick demo
    first_species = sorted(os.listdir(M03_P03_INPUT_TEST_FOLDER))[0]
    first_image = sorted(os.listdir(os.path.join(M03_P03_INPUT_TEST_FOLDER, first_species)))[0]
    test_img_path = os.path.join(M03_P03_INPUT_TEST_FOLDER, first_species, first_image)

    # print demo image information
    LOGGER("P07", f"test image:     {test_img_path}")
    LOGGER("P07", f"true species:   {first_species}")

    # run end-to-end prediction
    label, confidence = predictOneImage(test_img_path)

    # print predicted label and confidence
    LOGGER("P07", f"predicted:      {label}")
    LOGGER("P07", f"confidence:     {confidence*100:.2f}%")

    # compare prediction with true species name from the folder
    LOGGER("P07", f"correct:        {'YES ✓' if label == first_species else 'NO ✗'}")

    LOGGER("P07", "=== M03_P07 done ===")