from src.train.config import *
from src.models.CatDogClassifierV1 import CatDogClassifierV1
from src.loaders.DataLoader import CatVsDogsDataset
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from tqdm import tqdm
import logging
import os

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

def train(model, train_loader, test_loader, device, epochs, criterion, optimizer):
    """
    Train the model and evaluate on the test dataset after every epoch.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test dataset.
        device: The device to use for training (e.g., 'cuda' or 'cpu').
        epochs: Number of epochs to train.
        criterion: Loss function.
        optimizer: Optimizer for training.
    """

    print(f"TRAINING.... {TRAINING_ID}")
    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        # Training loop with progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), torch.unsqueeze(labels, -1).to(device) # Convert labels to float

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                predicted = (outputs > 0.5).long()  # Convert probabilities to binary predictions
                total += labels.size(0)
                correct += predicted.eq(labels.long()).sum().item()  # Ensure labels are long for comparison

                pbar.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

        # Log training results
        train_accuracy = 100. * correct / total
        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # Evaluate on the test dataset
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), torch.unsqueeze(labels, -1).to(device)  # Convert labels to float
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                predicted = (outputs > 0.5).long()  # Convert probabilities to binary predictions
                test_total += labels.size(0)
                test_correct += predicted.eq(labels.long()).sum().item()  # Ensure labels are long for comparison

        test_accuracy = 100. * test_correct / test_total
        logging.info(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
        print(f"Epoch {epoch+1}/{epochs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    model = CatDogClassifierV1().to(DEVICE)
    train_loader = DataLoader(CatVsDogsDataset(TRAIN_PATH, IMAGE_DIM), TRAIN_BATCH_SIZE, True)
    test_loader = DataLoader(CatVsDogsDataset(TEST_PATH, IMAGE_DIM), TEST_BATCH_SIZE, True)
    
    # Use BCEWithLogitsLoss for sigmoid activation
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train(model, train_loader, test_loader, DEVICE, EPOCHS, criterion, optimizer)

    # Save the model checkpoint
    checkpoint_path = os.path.join("checkpoints", f"{TRAINING_ID}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Model saved to {checkpoint_path}")