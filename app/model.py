from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
TEMPERATURE = 1.746
WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "resnet_ft.pt"

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model(path=WEIGHTS):
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()

@torch.no_grad()
def predict(model, img: Image.Image) -> dict:
    x = _preprocess(img.convert("RGB")).unsqueeze(0)
    probs = torch.softmax(model(x) / TEMPERATURE, dim=1)[0]
    return {c: round(p.item(), 4) for c, p in zip(CLASSES, probs)}
