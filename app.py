from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import io
import os
import gdown  # <-- Added for Google Drive support
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# Class labels for prediction
class_labels = [
    'Acne',
    'Actinic_Keratosis',
    'Basal_Cell_Carcinoma',
    'Eczema',
    'Fungal_Infections',
    'Melanoma',
    'Nevus'
]

# Define model architecture (must match training architecture)
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
    nn.Linear(512, len(class_labels))
)

# Model file path and Drive ID
MODEL_PATH = os.getenv("MODEL_PATH", "skinLesionModel.pth")
DRIVE_FILE_ID = "1ByKSpiYNm7l5_jpcdsytCK3L7QtmiOs1"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("🔽 Model file not found. Downloading from Google Drive...")
    try:
        gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
        print("✅ Downloaded model successfully.")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")

# Load model weights
try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    # Rename classifier keys (if needed)
    if 'classifier.9.weight' in state_dict:
        state_dict['classifier.8.weight'] = state_dict.pop('classifier.9.weight')
    if 'classifier.9.bias' in state_dict:
        state_dict['classifier.8.bias'] = state_dict.pop('classifier.9.bias')

    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded and ready for predictions.")
except Exception as e:
    print(f"❌ Failed to load model state dict: {e}")

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, 1)

        return jsonify({
            'prediction': class_labels[predicted.item()],
            'confidence': f"{conf.item() * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
