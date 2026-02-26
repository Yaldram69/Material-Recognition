import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image

# ===========================
# 1️⃣ Load Model
# ===========================
MODEL_PATH = "material_classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

categories = [
    "brick", "carpet", "ceramic", "fabric", "foliage", "food",
    "glass", "hair", "leather", "metal", "mirror", "other",
    "painted", "paper", "plastic", "polishedstone", "skin",
    "sky", "stone", "tile", "wallpaper","water", "wood"
]

# Load Model
print("🔄 Loading trained model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, len(categories))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("✅ Model Loaded Successfully!")

# ===========================
# 2️⃣ Image Preprocessing
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===========================
# 3️⃣ Real-time Webcam Testing
# ===========================
cap = cv2.VideoCapture(0)  # Open webcam
print("🎥 Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred_label = categories[torch.argmax(output).item()]

    cv2.putText(frame, f"Material: {pred_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Material Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()