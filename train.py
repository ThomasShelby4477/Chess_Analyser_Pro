import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Paths
data_dir = 'dataset'
model_save_path = 'chess_piece_model.pth'

# Training Config
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 13  # 12 pieces + empty
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# <<< --- THIS SECTION HAS BEEN CORRECTED --- >>>
# Transformations
# Using data augmentation to make the model more robust.
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),             # Randomly rotate images by up to 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Adjust image colors
    transforms.ToTensor(),
])
# <<< --- END CORRECTION --- >>>


# Dataset & DataLoader
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Label names (class index to name)
class_names = dataset.classes
print("Classes found:", class_names)
print(f"Training on {len(dataset)} images...")

# <<< --- THIS SECTION HAS BEEN CORRECTED --- >>>
# Load ResNet18 and modify output layer
# Using pre-trained weights (transfer learning) for much better accuracy.
model = models.resnet18(weights="IMAGENET1K_V1")
# <<< --- END CORRECTION --- >>>

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
loss_history = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")