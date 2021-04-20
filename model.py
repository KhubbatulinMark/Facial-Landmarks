import torchvision.models as models


def create_model():
    model = models.resnext101_32x8d(pretrained=True)
    model.fn = nn.Linear(model.fn.in_features, 2 * NUM_PTS, bias=True)

    return model
