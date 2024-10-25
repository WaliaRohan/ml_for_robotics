import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dubins_model import DubinsDataset, DubinsRNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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

# Main script for training
if __name__ == "__main__":

    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and prepare data loaders
    train_data_file = "data/train_data.npz"
    val_data_file = "data/val_data.npz"

    train_loader, val_loader = prepare_data_loaders(
        train_data_file = train_data_file, val_data_file = val_data_file)    

    # Initialize model
    input_size = 4  # x2, y2, psi, gamma
    hidden_size = 128
    output_size = 3  # Trajectory output: x, y, alt
    model = DubinsRNN(input_size, hidden_size, output_size).to(device)

    # Training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5

    # TensorBoard writer
    writer = SummaryWriter('runs/dubins_rnn_experiment_train_eval')

    # Record training time
    start_time = time.time()
    
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
    
        # Calculate elapsed time
    elapsed_time_seconds = time.time() - start_time

    # Convert to hours, minutes, and seconds
    elapsed_time = str(timedelta(seconds=elapsed_time_seconds))
    print("Training time: ", elapsed_time)

    # Save the trained model
    torch.save(model.state_dict(), 'dubins_rnn_model.pth')
    print("Model training complete and saved.")