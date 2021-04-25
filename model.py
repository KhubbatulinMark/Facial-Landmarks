import torch.nn as nn
import torchvision.models as models

from utils import NUM_PTS, CROP_SIZE


def create_model():
    model = models.densenet169(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True

    model.classifier = nn.Linear(model.classifier.in_features, 2 * NUM_PTS, bias=True)

    for param in model.classifier.parameters():
        param.requires_grad = True
    return model
