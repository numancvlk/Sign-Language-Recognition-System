#LIBRARIES
import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def getMaskModel(numClasses=29, device="cpu"):
    # En güncel ön-eğitimli ağırlıkları kullan
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    
    # Son sınıflandırma katmanını değiştir
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, numClasses)
    model = model.to(device)
    return model