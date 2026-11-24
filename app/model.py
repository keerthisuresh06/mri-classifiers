import os
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from huggingface_hub import hf_hub_download

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
TEMPERATURE = 1.746
WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "resnet_ft.pt"
MODEL_REPO = os.getenv("MODEL_REPO", "Keerthisuresh12/mri-resnet18")
MODEL_REVISION = os.getenv("MODEL_REVISION", "faaf6a12667e54bc9b972b80de10740f59bba5ad")

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def weights_path() -> Path:
    if WEIGHTS.exists():
        return WEIGHTS
    return Path(hf_hub_download(repo_id=MODEL_REPO, filename="resnet_ft.pt",
                                revision=MODEL_REVISION))

def load_model(path=None):
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(path or weights_path(), map_location="cpu"))
    return m.eval()

@torch.no_grad()
def predict(model, img: Image.Image) -> dict:
    x = _preprocess(img.convert("RGB")).unsqueeze(0)
    probs = torch.softmax(model(x) / TEMPERATURE, dim=1)[0]
    return {c: round(p.item(), 4) for c, p in zip(CLASSES, probs)}
