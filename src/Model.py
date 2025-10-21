#LIBRARIES
import torch
from torch import nn
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pretrainedModel = models.MobileNetV2(pretrained = True)

pretrainedModel.classifier[1] = nn.Linear(
    in_features= pretrainedModel.classifier[1].in_features,
    out_features=26
)

pretrainedModel = pretrainedModel.to(DEVICE)