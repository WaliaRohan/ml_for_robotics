import time

import numpy as np
import torch
import torch.nn as nn  # Import torch.nn for defining the RNN model
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from test_dubinEHF3d import \
    dubinEHF3d  # Import your Python ground truth function

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class
class DubinsDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]

# Function to pad or truncate trajectories to a fixed length
def pad_or_truncate_path(path, max_length=100):
    if len(path) < max_length:
        # Pad with zeros if the path is too short
        padding = np.zeros((max_length - len(path), 3))
        return np.vstack((path, padding))
    else:
        # Truncate if the path is too long
        return path[:max_length]

# Function to generate dataset based on the new grid and heading/climb configurations
def generate_large_dataset(grid_size=100, r_min=100, step_size=10):
    print(f"Generating large dataset...")

    # Placeholders for data
    input_data = []
    target_data = []

    # Grid of points (1000x1000 discretized into 100x100)
    x_values = np.arange(-500, 500, step_size)
    y_values = np.arange(-500, 500, step_size)
    
    # Headings from 0 to 350 degrees (36 values)
    heading_values = np.deg2rad(np.arange(0, 360, 10))

    # Climb angles from -30 to 30 degrees (12 values)
    climb_angles = np.deg2rad(np.arange(-30, 30, 5))

    max_length = 100  # Maximum trajectory length

    sample_idx = 0

    for x2 in x_values:
        for y2 in y_values:
            for psi in heading_values:
                for gamma in climb_angles:
                    # Call the Python function to generate the trajectory
                    step_length = 10
                    path, psi_end, num_path_points = dubinEHF3d(0, 0, 0, psi, x2, y2, r_min, step_length, gamma)

                    # Pad or truncate path to ensure consistent length
                    path_padded = pad_or_truncate_path(path, max_length)

                    # Store input and output
                    input_data.append([x2, y2, psi, gamma])  # Input: goal and parameters
                    target_data.append(path_padded)  # Output: padded/truncated trajectory

                    sample_idx += 1
                    if sample_idx % 10000 == 0:
                        print(f"{sample_idx} samples generated...")

    print("Large dataset generation complete!")
    return np.array(input_data), np.array(target_data)

# Prepare Data Loaders
def prepare_data_loaders(batch_size=64):
    inputs, targets = generate_large_dataset()
    dataset = DubinsDataset(inputs, targets)

    # Split dataset into train and validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Data loaders are ready!")
    return train_loader, val_loader

# RNN model
class DubinsRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(DubinsRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        return out

# Main script for training
if __name__ == "__main__":
    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders()

    # Initialize model
    input_size = 4  # x2, y2, psi, gamma
    hidden_size = 128
    output_size = 3  # Trajectory output: x, y, alt
    model = DubinsRNN(input_size, hidden_size, output_size).to(device)

    # Training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float().to(device).unsqueeze(1)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets[:, -1, :])
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'dubins_rnn_model.pth')

    print("Model training complete and saved.")