import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import random

# ===========================
# 1️⃣ Configuration
# ===========================
DATASET_PATH = "./dataset/minc-2500"
IMAGES_PATH = os.path.join(DATASET_PATH, "")
LABELS_PATH = os.path.join(DATASET_PATH, "labels")
MODEL_SAVE_PATH = "material_classifier.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 5  # Increase for better training
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {DEVICE}")

# ===========================
# 2️⃣ Load Category Labels
# ===========================
def load_categories(file_path):
    with open(file_path, "r") as f:
        categories = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(categories)} material categories.")
    return categories

categories = load_categories(os.path.join(DATASET_PATH, "categories.txt"))
category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

# ===========================
# 3️⃣ Load Dataset
# ===========================
def load_image_paths(label_file, base_path, sample_fraction=1.0):
    with open(label_file, "r") as f:
        paths = [os.path.join(base_path, line.strip()) for line in f.readlines()]
    valid_paths = [p for p in paths if os.path.exists(p)]
    return random.sample(valid_paths, max(1, int(len(valid_paths) * sample_fraction)))

train_images, val_images = [], []
for i in range(1, 3):
    train_images += load_image_paths(os.path.join(LABELS_PATH, f"train{i}.txt"), IMAGES_PATH)
    val_images += load_image_paths(os.path.join(LABELS_PATH, f"validate{i}.txt"), IMAGES_PATH)

print(f"📊 Train: {len(train_images)}, Val: {len(val_images)}")

# ===========================
# 4️⃣ PyTorch Dataset
# ===========================
class MINCDataset(Dataset):
    def __init__(self, image_paths, categories, transform=None):
        self.image_paths = image_paths
        self.categories = {category: idx for idx, category in enumerate(categories)}
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        category_name = os.path.basename(os.path.dirname(image_path))
        label = self.categories.get(category_name, -1)
        if self.transform:
            image = self.transform(image)
        return image, label

# Image Transformations
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MINCDataset(train_images, categories, transform=transform)
val_dataset = MINCDataset(val_images, categories, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📂 Datasets Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}")

# ===========================
# 5️⃣ Initialize Model
# ===========================
print("🔄 Initializing ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(categories))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ===========================
# 6️⃣ Training with Proper Logs
# ===========================
def train():
    print("🚀 Starting Training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"🟢 Epoch {epoch + 1}/{NUM_EPOCHS}")

        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{100 * correct / total:.2f}%")

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # TensorFlow-Like Logging
        print(f"📊 Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"loss: {avg_loss:.4f} - acc: {train_acc:.2f}% - "
              f"val_loss: {avg_val_loss:.4f} - val_acc: {val_acc:.2f}%")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()
