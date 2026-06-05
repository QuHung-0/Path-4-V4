# ========= FILE NAME: M04_P01_runModelOnAllTestImagesAndCollectPredictions.py =========
# FILE ROLE: Feeds all test images through the trained model and saves every prediction to CSV
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P01_F01 → P01_F03

# import OS module for file and folder path handling
import os

# import CSV module for writing prediction results
import csv

# import PyTorch core
import torch

# import everything M04_P01 needs from the shared input paths file
from model_weights.M04_evaluation.input.M04_I01_inputPaths import (
    M04_P01_INPUT_TEST_FOLDER,          # folder containing the test dataset
    M04_P01_INPUT_NORM_STATS_JSON,      # normalization stats file
    M04_P01_INPUT_BEST_MODEL_PTH,       # best trained model weights
    M04_P01_INPUT_CLASS_TO_IDX_JSON,    # class-to-index mapping file

    M04_BATCH_SIZE,                    # batch size for inference

    M04_P01_OUTPUT_PREDICTIONS_CSV,    # output CSV path for saved predictions
    M04_INPUT_IMAGE_SIZE,              # image size used by the model

    LOGGER,                            # logger

    P03_F08_createImageFolderDatasetFromPath,             # create dataset from folder
    P03_F10_wrapDatasetInDataloaderWithBatchSizeAndShuffle,  # wrap dataset in DataLoader
    P03_F11_loadClassToIndexMappingFromJsonFile,          # load idx→class mapping

    P07_F23_loadTrainedModelWeightsFromFile,              # load trained model from disk

    P02_F04_loadNormalizationStatsFromJsonFile,           # load mean/std
    P02_F06_buildValAndTestTransformNoAugmentation,       # build inference-style transform
)


# FUNCTION 1
def P01_F01_loadTestDatasetWithNoAugmentation(test_folder, transform):
    # wraps the test folder in an ImageFolder dataset — no shuffling, no augmentation

    # build dataset from folder structure
    dataset = P03_F08_createImageFolderDatasetFromPath(test_folder, transform)

    # wrap dataset in a DataLoader with shuffle disabled so prediction order stays stable
    loader = P03_F10_wrapDatasetInDataloaderWithBatchSizeAndShuffle(
        dataset, M04_BATCH_SIZE, shuffle=False
    )

    # return both the dataset and loader because later code needs both the images and file paths
    return dataset, loader


# FUNCTION 2
def P01_F02_runInferenceOnEntireTestSetAndReturnResults(
    model, dataset, loader, idx_to_class: dict, device: torch.device
) -> list:
    # feeds every batch through the model
    # also records filename and full image_path for each prediction
    # this lets later modules copy or inspect the exact file that was predicted

    # put model into evaluation mode
    model.eval()

    # list of per-image result dictionaries
    results = []

    # dataset.samples is a list of (full_path, class_index) pairs in the same order as the dataset
    # because shuffle=False, the dataloader will preserve this order
    all_image_paths = [path for path, _ in dataset.samples]

    # tracks which sample path corresponds to the next prediction
    image_index = 0

    # disable gradients because this is inference only
    with torch.no_grad():

        # loop through test batches
        for images, labels in loader:

            # move batch to the selected compute device
            images = images.to(device)

            # run the model
            outputs = model(images)

            # convert logits to probabilities
            probs = torch.softmax(outputs, dim=1)

            # find best class and confidence for each image in the batch
            confs, preds = probs.max(dim=1)

            # unpack each item in the batch one by one
            for i in range(len(labels)):

                # get the original full path of this image
                full_path = all_image_paths[image_index]
                image_index += 1

                # save a result record for later reporting
                results.append({
                    "filename": os.path.basename(full_path),                 # file name only
                    "image_path": full_path,                                 # full absolute/relative path
                    "true_label": idx_to_class[labels[i].item()],            # ground-truth species name
                    "predicted_label": idx_to_class[preds[i].item()],        # predicted species name
                    "confidence": round(confs[i].item(), 6),                 # confidence score
                    "correct": int(labels[i].item() == preds[i].item()),     # 1 if correct, 0 if wrong
                })

    LOGGER("P01", f"inference done — {len(results)} predictions collected")
    return results


# FUNCTION 3
def P01_F03_saveAllPredictionsAndTrueLabelsToCSV(results: list, output_csv: str) -> None:
    # writes one row per image — includes filename and image_path columns

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # open CSV file for writing
    with open(output_csv, "w", newline="", encoding="utf-8") as f:

        # define columns in the output file
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "image_path",
            "true_label", "predicted_label", "confidence", "correct"
        ])

        # write header row
        writer.writeheader()

        # write all prediction rows
        writer.writerows(results)

    LOGGER("P01", f"predictions CSV saved → {output_csv}")


# MAIN
if __name__ == "__main__":
    # direct-run entry point for evaluating the model on the full test set

    LOGGER("P01", "=== M04_P01 — running model on all test images ===")

    # choose GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER("P01", f"device: {device}")

    LOGGER("P01", "step 1 — loading transform, dataset, model...")

    # load normalization statistics
    mean, std = P02_F04_loadNormalizationStatsFromJsonFile(M04_P01_INPUT_NORM_STATS_JSON)

    # build evaluation transform
    transform = P02_F06_buildValAndTestTransformNoAugmentation(mean, std, M04_INPUT_IMAGE_SIZE)

    # build test dataset and loader
    dataset, loader = P01_F01_loadTestDatasetWithNoAugmentation(M04_P01_INPUT_TEST_FOLDER, transform)

    # load index-to-class mapping
    idx_to_class = P03_F11_loadClassToIndexMappingFromJsonFile(M04_P01_INPUT_CLASS_TO_IDX_JSON)

    # load trained model weights
    model = P07_F23_loadTrainedModelWeightsFromFile(M04_P01_INPUT_BEST_MODEL_PTH, device)

    LOGGER("P01", "step 2 — running inference on all test images...")

    # run inference over the complete test set
    results = P01_F02_runInferenceOnEntireTestSetAndReturnResults(
        model, dataset, loader, idx_to_class, device
    )

    # compute overall accuracy from results
    total = len(results)
    correct = sum(r["correct"] for r in results)
    LOGGER("P01", f"overall accuracy: {correct}/{total} = {100*correct/total:.2f}%")

    LOGGER("P01", "step 3 — saving predictions to CSV...")

    # write results to CSV for later metric calculations
    P01_F03_saveAllPredictionsAndTrueLabelsToCSV(results, M04_P01_OUTPUT_PREDICTIONS_CSV)

    LOGGER("P01", "=== M04_P01 done ===")