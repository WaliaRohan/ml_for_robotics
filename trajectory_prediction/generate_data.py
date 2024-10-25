import gc

import numpy as np
from dubinEHF3d import dubinEHF3d
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split


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
    print("Total possible samples: ", total_samples)

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

    print("Total samples generated: ", sample_idx+1)
    print("Large dataset generation complete!")
    return np.array(input_data), np.array(target_data)

# Main script for training
if __name__ == "__main__":

    inputs, targets = generate_large_dataset(1000, 100, 10)
    
    inputs_train, inputs_temp, targets_train, targets_temp = train_test_split(inputs, targets, test_size=0.2, random_state=42)
    np.savez('train_data.npz', inputs_train=inputs_train, targets_train=targets_train)
    del inputs, targets, inputs_train, targets_train
    gc.collect() # Manually trigger garbage collection
    
    inputs_val, inputs_test, targets_val, targets_test = train_test_split(inputs_temp, targets_temp, test_size=0.5, random_state=42)
    np.savez('val_data.npz', inputs_val=inputs_val, targets_val=targets_val)
    np.savez('test_data.npz', inputs_test=inputs_test, targets_test=targets_test)
