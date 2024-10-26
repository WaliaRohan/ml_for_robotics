import time
from datetime import timedelta

import matplotlib.pyplot as plt
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
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    num_epochs = 1
    train_losses = []
    val_losses = []

    # TensorBoard writer
    writer = SummaryWriter('runs/dubins_rnn_experiment_train_eval')

    # Record training time
    start_time = time.time()

    # Create arrays for training and validation losses
    
    for epoch in range(num_epochs):
        
        # Training phase
        model.train()
        train_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            inputs = inputs.unsqueeze(1).repeat(1, 100, 1)
            targets = targets.float().to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Log training loss to TensorBoard
            train_loss += loss.item()
            writer.add_scalar('Training Loss', train_loss / len(train_loader), epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0

        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.float().to(device).unsqueeze(1)
            targets = targets.float().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()

            # Log validation loss to TensorBoard
            writer.add_scalar('Validation Loss', val_loss / len(val_loader), epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {train_loss / len(train_loader):.4f}")
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

    # Calculate elapsed time
    elapsed_time_seconds = time.time() - start_time
    elapsed_time = str(timedelta(seconds=elapsed_time_seconds)) # HH-MM-SS
    print("Training/Validation time: ", elapsed_time)

    # Save the trained model
    torch.save(model.state_dict(), 'dubins_rnn_model.pth')
    print("Model training complete and saved.")

    # Save training and validation losses
    np.savez('train_val_losses.npz', train_losses=train_losses, val_losses=val_losses, num_epochs=num_epochs)

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    
    # Set y-axis range and increments
    plt.ylim(0, 100)
    # plt.yticks(range(0, 101, 5e))  # Y-axis from 0 to 100 in steps of 5

    # Set x-axis to show only integer values
    plt.xticks(range(1, num_epochs + 1))

    # Add legend in the top right corner
    plt.legend(loc='upper right')
    plt.grid()

    # Display the plot
    plt.show()

    # Save the plot
    plt.savefig('training_validation_loss.png')