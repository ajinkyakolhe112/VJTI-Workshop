import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# TODO: Stage 1 Dataset
# Load MNIST dataset from Hugging Face
def load_mnist():
    # Load the dataset
    dataset = load_dataset("mnist")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Convert datasets to PyTorch format
    train_dataset = dataset["train"].with_transform(
        lambda x: {"pixel_values": transform(x["image"]), "label": x["label"]}
    )
    test_dataset = dataset["test"].with_transform(
        lambda x: {"pixel_values": transform(x["image"]), "label": x["label"]}
    )
    
    return train_dataset, test_dataset

# TODO: Stage 2 Model
# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# TODO: Stage 3 Training the Model
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluate_model(model, test_loader, device)
    
    # Save the model
    torch.save(model.state_dict(), "mnist_model.pth")
    print("\nModel saved as 'mnist_model.pth'")

if __name__ == "__main__":
    main() 