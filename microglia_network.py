import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MicrogliaDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class MicrogliaNetwork(nn.Module):
    def __init__(self, input_size):
        super(MicrogliaNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def main():
    # Load data
    print("Loading data...")
    data = pd.read_csv('data.csv')
    
    # Prepare features (excluding Sample Name column)
    X = data.drop('Sample Name', axis=1).values
    
    # For this example, we'll predict the Mean_Fluorescent_Intensity_of_CD11b_Average
    # You can modify this target based on your specific needs
    target_column = 'Mean_Fluorescent_Intensity_of_CD11b_Average'
    y = data[target_column].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create data loaders
    train_dataset = MicrogliaDataset(X_train_scaled, y_train)
    test_dataset = MicrogliaDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Initialize model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = MicrogliaNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("\nTraining the neural network...")
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer)
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_predictions = []
        test_actual = []
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            test_predictions.extend(outputs.numpy().flatten())
            test_actual.extend(batch_y.numpy())
    
    # Calculate and print metrics
    mse = np.mean((np.array(test_predictions) - np.array(test_actual)) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((np.array(test_actual) - np.array(test_predictions)) ** 2) / 
              np.sum((np.array(test_actual) - np.mean(np.array(test_actual))) ** 2))
    
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(test_actual, test_predictions, alpha=0.5)
    plt.plot([min(test_actual), max(test_actual)], [min(test_actual), max(test_actual)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    plt.savefig('predictions_vs_actual.png')
    plt.close()

if __name__ == "__main__":
    main()
