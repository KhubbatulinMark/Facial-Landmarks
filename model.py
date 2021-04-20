import torch.nn as nn
import torchvision.models as models

from utils import NUM_PTS, CROP_SIZE


def create_model():
    model = models.resnext101_32x8d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)

    return model
