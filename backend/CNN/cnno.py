import os
import cv2
import numpy as np
import random
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# ==============================================================
# 1) DATASET DEFINITION
# ==============================================================

class EyeStateDataset(Dataset):
    """
    A dataset for eye state classification.
    Expects two directories: one for open eyes and one for closed eyes.
    """
    def __init__(self, open_dir, closed_dir, img_size=(299, 299), transform=None):
        self.image_paths = []
        self.labels = []  # 0: open, 1: closed
        self.img_size = img_size
        self.transform = transform

        # Load open eye images (label 0)
        for path in glob(os.path.join(open_dir, '*.jpg')):
            self.image_paths.append(path)
            self.labels.append(0)
        # Load closed eye images (label 1)
        for path in glob(os.path.join(closed_dir, '*.jpg')):
            self.image_paths.append(path)
            self.labels.append(1)

        # Shuffle dataset
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image using cv2 (OpenCV loads images in BGR)
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to the desired input size
        img = cv2.resize(img, self.img_size)

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)
        else:
            # Convert to tensor and normalize to [0,1]
            img = transforms.ToTensor()(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

# ==============================================================
# 2) MODEL DEFINITION: CNN FOR EYE STATE CLASSIFICATION
# ==============================================================

class EyeBlinkCNN(nn.Module):
    def __init__(self):
        super(EyeBlinkCNN, self).__init__()
        # Load InceptionV3 with pretrained weights.
        # Note: using the new weights API; this will automatically set aux_logits=True.
        self.cnn = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        # Replace the final fc layer with an identity so we can add our own classifier.
        self.cnn.fc = nn.Identity()
        
        # Optionally freeze the feature extractor layers:
        # for param in self.cnn.parameters():
        #     param.requires_grad = False

        # Define your custom classifier.
        # InceptionV3 (with fc replaced) outputs a 2048-dimensional feature vector.
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Outputs a probability between 0 and 1
        )
        
    def forward(self, x):
        """
        Forward pass.
          x: input tensor of shape (batch, 3, 299, 299)
        """
        # In training mode, inception_v3 returns a tuple: (main_output, aux_output)
        features = self.cnn(x)
        if isinstance(features, tuple):
            # Use the main output only.
            features = features[0]
        # Now pass through your classifier.
        out = self.classifier(features)
        return out

# ==============================================================
# 3) TRAINING & EVALUATION FUNCTIONS
# ==============================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device)               # (batch, C, H, W)
        labels = labels.to(device).unsqueeze(1)  # (batch, 1)

        optimizer.zero_grad()
        outputs = model(imgs)                # (batch, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        # Use threshold of 0.5: below 0.5 means open (label 0), otherwise closed (label 1)
        preds = (outputs >= 0.5).float()  # Prediction: 1 = closed, 0 = open
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ==============================================================
# 4) BLINK DETECTION FROM A SEQUENCE OF PREDICTIONS
# ==============================================================

def detect_blinks(predictions, threshold=0.5, min_closed_frames=1):
    """
    Given a list of probabilities for each frame, detect blinks.
    A blink is defined as an open (prob < threshold) -> one or more closed (prob >= threshold) -> open transition.
    :param predictions: List of probabilities from the CNN.
    :param threshold: Probability threshold to decide open vs closed.
    :param min_closed_frames: Minimum consecutive frames with closed state to count as a blink.
    :return: Number of detected blinks.
    """
    # Convert probabilities to binary states:
    # 0: open, 1: closed
    states = [1 if p >= threshold else 0 for p in predictions]

    blink_count = 0
    i = 0
    while i < len(states) - 2:
        if states[i] == 0:  # frame is open
            j = i + 1
            closed_count = 0
            # Count consecutive closed frames
            while j < len(states) and states[j] == 1:
                closed_count += 1
                j += 1
            if closed_count >= min_closed_frames and j < len(states) and states[j] == 0:
                blink_count += 1
                i = j  # Skip ahead past this blink
                continue
        i += 1
    return blink_count

# ==============================================================
# 5) MAIN SCRIPT
# ==============================================================

if __name__ == "__main__":
    # Paths to your datasets
    open_dir = "/Users/waasiqahmed/Desktop/eye_data/OpenFace"     
    closed_dir = " /Users/waasiqahmed/Desktop/eye_data/ClosedFace"


    # Define transforms: converting to tensor and normalizing using ImageNet stats.
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create the dataset and split into training and test sets (e.g., 80/20 split)
    full_dataset = EyeStateDataset(open_dir, closed_dir, img_size=(299, 299), transform=data_transform)
    total_size = len(full_dataset)
    test_size = int(0.2 * total_size)
    train_size = total_size - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss, and optimizer
    model = EyeBlinkCNN().to(device)
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 15
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the model state
    torch.save(model.state_dict(), "eye_blink_cnn.pth")
    print("Model saved as eye_blink_cnn.pth")
    
    # ==========================================================
    # EXAMPLE: BLINK DETECTION ON A SEQUENCE OF IMAGES
    # ==========================================================
    # Suppose you have a video or sequence of images from which you extract frames.
    # Here we simulate this with a list of image file paths (you can adjust this as needed).
   #
   # test_sequence = sorted(glob("/path/to/test_sequence/*.jpg"))
    #
   # sequence_predictions = []
   # model.eval()
   # with torch.no_grad():
     #   for img_path in test_sequence:
    #        # Load image in RGB format
     #       img = cv2.imread(img_path)
      #      if img is None:
     #           continue
      #      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      #      img = cv2.resize(img, (299, 299))
            # Apply the same transform used in training
     #       img_tensor = data_transform(img).unsqueeze(0).to(device)  # shape: (1, 3, 299, 299)
       #     output = model(img_tensor)
            # Get probability as a float (output is of shape (1,1))
      ##      prob = output.item()
         #   sequence_predictions.append(prob)

    # Now detect blinks in the sequence
  #  num_blinks = detect_blinks(sequence_predictions, threshold=0.5, min_closed_frames=1)
  #  print("Detected blinks in sequence:", num_blinks)