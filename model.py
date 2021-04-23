import torch.nn as nn
import torchvision.models as models

from utils import NUM_PTS, CROP_SIZE


def create_model():
    model = models.resnet101(pretrained=True)
    model.requires_grad_(True)

    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.fc.requires_grad_(True)

    return model
