import torch
from torchvision import transforms, models
from PIL import Image

# Load the same model architecture
NUM_CLASSES = 13
model = models.resnet18(weights=None)  # or weights="IMAGENET1K_V1" if you used pretrained during training
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)

# Load weights
model.load_state_dict(torch.load("chess_piece_model.pth", map_location=DEVICE))
model.eval()

# Transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Class names (match the folder order in dataset)
class_names = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'empty', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']  # Adjust if needed

# Inference
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        pred_idx = torch.argmax(output, 1).item()

    return class_names[pred_idx]

# Example usage
img_path = "dataset/bK/0_1.png"  # Change to your image
prediction = predict(img_path)
print("Prediction:", prediction)
