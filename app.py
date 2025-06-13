# app.py  – Flask API for skin‑lesion classification
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import io, os, gdown

# --------------------------------------------------
# 1. Config & one‑time model download
# --------------------------------------------------
CLASS_LABELS = [
    "Acne", "Actinic_Keratosis", "Basal_Cell_Carcinoma",
    "Eczema", "Fungal_Infections", "Melanoma", "Nevus"
]

MODEL_URL  = os.getenv(
    "MODEL_URL",
    "https://drive.google.com/uc?export=download&id=1ByKSpiYNm7l5_jpcdsytCK3L7QtmiOs1"
)
MODEL_PATH = os.getenv("MODEL_PATH", "skinLesionModel.pth")

def download_model(url=MODEL_URL, dst=MODEL_PATH):
    """Download the .pth file from Google Drive if it's not already present."""
    if os.path.exists(dst):
        return dst
    print("⬇️  Downloading model weights from Google Drive …")
    gdown.download(url, dst, quiet=False, fuzzy=True)
    print("✅  Model downloaded to", dst)
    return dst

# Make sure weights exist before building the network
download_model()

# --------------------------------------------------
# 2. Build network & load weights
# --------------------------------------------------
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 1024),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(1024),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(512),
    nn.Linear(512, len(CLASS_LABELS))
)

try:
    state = torch.load(MODEL_PATH, map_location="cpu")
    # rename keys if the training script saved them with .9.*
    if "classifier.9.weight" in state:
        state["classifier.8.weight"] = state.pop("classifier.9.weight")
        state["classifier.8.bias"]   = state.pop("classifier.9.bias")
    model.load_state_dict(state)
    model.eval()
    print("🪄  Model loaded and ready.")
except Exception as e:
    print("❌  Could not load model:", e)

# --------------------------------------------------
# 3. Flask setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)
            conf, idx = torch.max(probs, 1)

        return jsonify({
            "prediction": CLASS_LABELS[idx.item()],
            "confidence": f"{conf.item()*100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# 4. Entry‑point
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
