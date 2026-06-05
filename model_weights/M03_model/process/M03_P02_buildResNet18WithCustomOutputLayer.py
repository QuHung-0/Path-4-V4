# ========= FILE NAME: M03_P02_buildResNet18WithCustomOutputLayer.py =========
# FILE ROLE: Loads pretrained ResNet-18 and replaces the final layer for 10 fish species
# TOTAL FUNCTIONS IN THIS FILE: 4
# FUNCTION INDEX RANGE: P02_F04 → P02_F07

# import PyTorch core
import torch

# import neural-network layers module
import torch.nn as nn

# import torchvision model zoo
from torchvision import models

# import the ResNet-18 weight preset enum
from torchvision.models import ResNet18_Weights

# import project settings and logger
from model_weights.M03_model.input.M03_I01_inputPaths import (
    RANDOM_SEED,     # global seed value
    M03_NUM_CLASSES,  # number of fish classes
    LOGGER            # logging function
)


# FUNCTION 4
def P02_F04_loadPretrainedResNet18FromPytorchHub() -> nn.Module:
    # downloads ResNet-18 weights trained on ImageNet (1.28 million images, 1000 classes)
    # these pretrained weights mean the model already knows edges, textures, shapes

    LOGGER("P02", "loading ResNet-18 with pretrained ImageNet weights...")

    # load the official pretrained ResNet-18 backbone
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # report success
    LOGGER("P02", "pretrained ResNet-18 loaded — 11 million parameters from ImageNet")

    return model


# FUNCTION 5
def P02_F05_replaceLastFullyConnectedLayerWithNOutputs(
    model: nn.Module, n_classes: int
) -> nn.Module:
    # ResNet-18 ends with Linear(512, 1000) for 1000 ImageNet classes
    # we cut that off and put Linear(512, n_classes) so it outputs one score per fish species

    # read the number of features feeding into the final classifier
    in_features = model.fc.in_features

    # replace the classifier head with a new linear layer sized for this dataset
    model.fc = nn.Linear(in_features, n_classes)

    # log the change for traceability
    LOGGER("P02", f"replaced final layer: Linear(512, 1000) → Linear({in_features}, {n_classes})")

    return model


# FUNCTION 6
def P02_F06_printModelSummaryAndTotalParameterCount(model: nn.Module) -> None:
    # counts total trainable parameters and prints a brief summary

    # count every parameter in the network
    total_params = sum(p.numel() for p in model.parameters())

    # count only parameters that will actually be updated during training
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print summary information
    LOGGER("P02", f"total parameters:     {total_params:,}")
    LOGGER("P02", f"trainable parameters: {trainable_params:,}")
    LOGGER("P02", f"final layer output:   {model.fc.out_features} classes")


# FUNCTION 7
def P02_F07_runOneForwardPassWithFakeImageToVerifyShapes(
    model: nn.Module, device: torch.device
) -> None:
    # creates a fake 1-image batch, runs it through the model, confirms output shape is [1, 10]

    # switch model to evaluation mode so batch norm / dropout behave deterministically
    model.eval()

    # create a dummy input tensor of shape [batch, channels, height, width]
    fake_input = torch.zeros(1, 3, 224, 224).to(device)

    # disable gradient tracking because this is only a shape test
    with torch.no_grad():
        output = model(fake_input)

    # log the input and output shapes
    LOGGER("P02", f"forward pass check — input: {list(fake_input.shape)}  →  output: {list(output.shape)}")

    # assert that the output size matches the number of classes expected by the project
    assert output.shape == (1, M03_NUM_CLASSES), f"expected output shape (1, {M03_NUM_CLASSES}), got {output.shape}"

    # confirm that the architecture is wired correctly
    LOGGER("P02", "shape verified ✓ — model is ready for training")


# MAIN
if __name__ == "__main__":

    # import the GPU/device and seed helpers from M03_P01
    from M03_P01_checkGpuAndSetReproducibilitySeed import (
        P01_F01_checkIfGpuIsAvailableAndPrintItsName,
        P01_F02_setAllRandomSeedsForFullReproducibility,
    )

    LOGGER("P02", "=== M03_P02 — building ResNet-18 ===")

    # detect device again so this file can run independently
    device = P01_F01_checkIfGpuIsAvailableAndPrintItsName()

    # set all random seeds again for reproducibility
    P01_F02_setAllRandomSeedsForFullReproducibility(RANDOM_SEED)

    # load pretrained ResNet-18
    model = P02_F04_loadPretrainedResNet18FromPytorchHub()

    # replace final classifier layer so the network predicts fish species instead of ImageNet classes
    model = P02_F05_replaceLastFullyConnectedLayerWithNOutputs(model, M03_NUM_CLASSES)

    # move model to GPU if available, otherwise CPU
    model = model.to(device)

    # print parameter summary
    P02_F06_printModelSummaryAndTotalParameterCount(model)

    # run a fake forward pass to verify tensor shapes
    P02_F07_runOneForwardPassWithFakeImageToVerifyShapes(model, device)

    LOGGER("P02", "=== M03_P02 done ===")