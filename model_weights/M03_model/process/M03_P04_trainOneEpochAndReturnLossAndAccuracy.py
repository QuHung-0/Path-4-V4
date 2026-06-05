# ========= FILE NAME: M03_P04_trainOneEpochAndReturnLossAndAccuracy.py =========
# FILE ROLE: Runs one full pass through training data and one full pass through val data
# TOTAL FUNCTIONS IN THIS FILE: 3
# FUNCTION INDEX RANGE: P04_F12 → P04_F14

# import OS module for folder creation when saving model weights
import os

# import PyTorch core library
import torch

# import PyTorch neural-network module for model types and loss functions
import torch.nn as nn

# import the project logger from the shared input config
from model_weights.M03_model.input.M03_I01_inputPaths import (
    LOGGER  # used to print progress and save-status messages
)


# FUNCTION 12
def P04_F12_runOneTrainingEpochAndUpdateWeights(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> tuple:
    # sets model to training mode, feeds every batch through, updates weights after each batch
    # returns (average_loss, accuracy_percent) for this epoch

    # put model into training mode so layers like dropout and batchnorm behave as training layers
    model.train()

    # total loss accumulator across all images in the epoch
    total_loss = 0.0

    # count how many predictions were correct
    total_correct = 0

    # count how many images were processed in total
    total_images = 0

    # loop through the training dataloader batch by batch
    for images, labels in train_loader:

        # move images to GPU or CPU depending on selected device
        images = images.to(device)

        # move labels to same device
        labels = labels.to(device)

        # clear old gradients from the previous batch
        optimizer.zero_grad()

        # run forward pass through the model
        outputs = model(images)

        # compute the loss between predicted logits and true labels
        loss = loss_fn(outputs, labels)

        # run backpropagation to compute gradients
        loss.backward()

        # update model weights using the optimizer
        optimizer.step()

        # add this batch's loss weighted by number of images
        total_loss += loss.item() * images.size(0)

        # choose predicted class as the one with the highest logit
        predicted = outputs.argmax(dim=1)

        # count how many predictions matched the ground-truth labels
        total_correct += (predicted == labels).sum().item()

        # add the number of images in this batch
        total_images += images.size(0)

    # compute average loss over the whole epoch
    avg_loss = total_loss / total_images

    # compute accuracy percentage over the whole epoch
    accuracy = 100.0 * total_correct / total_images

    # return average loss and accuracy for logging
    return avg_loss, accuracy


# FUNCTION 13
def P04_F13_runOneValidationEpochWithNoGradients(
    model: nn.Module,
    val_loader,
    loss_fn: nn.Module,
    device: torch.device
) -> tuple:
    # sets model to eval mode, runs validation images through, does NOT update weights
    # returns (average_loss, accuracy_percent) for this validation pass

    # switch to evaluation mode so dropout/batchnorm behave deterministically
    model.eval()

    # accumulators for total validation loss, correct predictions, and image count
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    # disable gradient tracking because validation does not train the model
    with torch.no_grad():

        # loop through validation batches
        for images, labels in val_loader:

            # move batch to chosen device
            images = images.to(device)
            labels = labels.to(device)

            # forward pass only
            outputs = model(images)

            # compute validation loss
            loss = loss_fn(outputs, labels)

            # accumulate weighted loss
            total_loss += loss.item() * images.size(0)

            # predict class with highest score
            predicted = outputs.argmax(dim=1)

            # count correct predictions
            total_correct += (predicted == labels).sum().item()

            # count images seen
            total_images += images.size(0)

    # compute average validation loss
    avg_loss = total_loss / total_images

    # compute validation accuracy percentage
    accuracy = 100.0 * total_correct / total_images

    # return validation summary
    return avg_loss, accuracy


# FUNCTION 14
def P04_F14_saveBestModelWeightsWhenValLossImproves(
    model: nn.Module,
    val_loss: float,
    best_loss: float,
    save_path: str
) -> float:
    # compares current val loss to best seen so far — saves weights if this is a new best
    # returns the new best loss value

    # only save if current validation loss is lower than the previous best
    if val_loss < best_loss:

        # create the destination folder if it does not already exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # save model parameters only, not the full model object
        torch.save(model.state_dict(), save_path)

        # log that a new best model was found
        LOGGER("P04", f"  ✓ new best model saved (val_loss {val_loss:.4f} < {best_loss:.4f})")

        # return updated best loss
        return val_loss

    # if no improvement, keep the old best loss
    return best_loss