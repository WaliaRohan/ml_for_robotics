import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn  # Import torch.nn for defining the RNN model
import torch.optim as optim
from dubinEHF3d import dubinEHF3d
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

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
def generate_large_dataset(grid_size=1000, r_min=100, step_size=10):
    print(f"Generating large dataset...")

    # Placeholders for data
    input_data = []
    target_data = []

    # Grid of points
    x_values = np.arange(-grid_size/2, grid_size/2, step_size)
    y_values = np.arange(-grid_size/2, grid_size/2, step_size)
    
    # Headings from 0 to 350 degrees (36 values)
    heading_values = np.deg2rad(np.arange(0, 360, 10))

    # Climb angles from -30 to 30 degrees (12 values)
    climb_angles = np.deg2rad(np.arange(-30, 30, 5))

    max_length = 100  # Maximum trajectory length

    sample_idx = 0

    total_samples = len(x_values)*len(y_values)*len(heading_values)*len(climb_angles)
    print("Total samples to be generated: ", total_samples)

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
def prepare_data_loaders(train_data_file, val_data_file, batch_size=64):

    inputs_train  = np.load(train_data_file)['inputs_train']
    targets_train = np.load(train_data_file)['targets_train']

    inputs_val  = np.load(val_data_file)['inputs_val']
    targets_val = np.load(val_data_file)['targets_val']
    
    dataset_train = DubinsDataset(inputs_train, targets_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = DubinsDataset(inputs_val, targets_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    print("Data loaders are ready!")
    return dataloader_train, dataloader_val

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


    train_data_file = "train_data.npz"
    val_data_file = "val_data.npz"

    train_loader, val_loader = prepare_data_loaders(
        train_data_file = train_data_file, val_data_file = val_data_file)    

    # Initialize model
    input_size = 4  # x2, y2, psi, gamma
    hidden_size = 128
    output_size = 3  # Trajectory output: x, y, alt
    model = DubinsRNN(input_size, hidden_size, output_size).to(device)

    # TensorBoard writer
    writer = SummaryWriter('runs/dubins_rnn_experiment')

    # Training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    num_epochs = 5
    for epoch in range(num_epochs):

        # Training phase
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

            # Log training loss to TensorBoard
            writer.add_scalar('Training Loss', train_loss / len(train_loader), epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

        # # Validation phase
        # model.eval()
        # val_loss = 0.0

    
        # Calculate elapsed time
    elapsed_time_seconds = time.time() - start_time

    # Convert to hours, minutes, and seconds
    elapsed_time = str(timedelta(seconds=elapsed_time_seconds))
    print("Training time: ", elapsed_time)

    # Save the trained model
    torch.save(model.state_dict(), 'dubins_rnn_model.pth')
    print("Model training complete and saved.")


# Pad or truncate: When we truncate the path, should the "goal" be the original goal, or should it be the original value or the final value?

# How do we identify vanishing gradients? Is how we did it correct? -> https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html 
# Average of norm of grad of all the weights

# Tips to improve:

# 1. Implement dropout
# 2. Try multi-layer LSTM ("n_layers" is the number of LSTM "cells")
# 3. Try xavier initialization
# 4. Plot the output trajectoties
# 5. Understand truncation
# 6. Weight Decay: reduce_lr_on_plateau (reduce learning rate on plateaus) https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html 
# 7. Can try a smaller learning rate
# 8. Increase number of epochs