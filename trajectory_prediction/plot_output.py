import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

from dubinEHF3d import dubinEHF3d
from dubins_model import DubinsRNN


def plot_dubin_paths(ground_truth_trajectories, predicted_trajectories):
    # Loop over each pair of ground truth and predicted trajectory
    for i, (gt_path, pred_path) in enumerate(zip(ground_truth_trajectories, predicted_trajectories)):
        fig = plt.figure(figsize=(8, 6), constrained_layout = True)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot ground truth path in blue
        ax.plot(gt_path[:, 0], gt_path[:, 1], gt_path[:, 2], 'b.-', label='Ground Truth')
        ax.plot([gt_path[0, 0]], [gt_path[0, 1]], [gt_path[0, 2]], 'r*')  # Start point for ground truth
        ax.plot(gt_path[-1, 0], gt_path[-1, 1], gt_path[-1, 2], 'bo')  # End point for ground truth

        # Plot predicted path in green
        ax.plot(pred_path[:, 0], pred_path[:, 1], pred_path[:, 2], 'g.-', label='Predicted')
        ax.plot(pred_path[-1, 0], pred_path[-1, 1], pred_path[-1, 2], 'go')  # End point for predicted

        # Set labels, grid, and legend
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('alt')
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Set title for each plot
        ax.set_title(f'Trajectory Plot {i+1}')

        plt.savefig(f'plots/Trajectory Comparison {i+1}.png')
    
    # Show all figures at once
    plt.show()


def generate_random_ground_truths():
    # Generate data values for assessing output
    grid_size=1000
    r_min=100
    step_size=10

    # Grid of points
    x_values = np.arange(-grid_size/2, grid_size/2, step_size)
    y_values = np.arange(-grid_size/2, grid_size/2, step_size)

    # Grid of points
    x_values = np.arange(-grid_size/2, grid_size/2, step_size)
    y_values = np.arange(-grid_size/2, grid_size/2, step_size)
    
    # Headings from 0 to 350 degrees (36 values)
    heading_values = np.deg2rad(np.arange(0, 360, 10))

    # Climb angles from -30 to 30 degrees (12 values)
    climb_angles = np.deg2rad(np.arange(-30, 30, 5)) 

    trajectories = []
    inputs = []

    x1 = 0
    y1 = 0
    alt1 = 0

    steplength = 10

    while len(trajectories) < 10:
        x2 = np.random.choice(x_values)
        y2 = np.random.choice(y_values)
        psi = np.random.choice(heading_values)
        gamma = np.random.choice(climb_angles)

        # Run dubinEHF3d with sampled values
        path, psi_end, num_path_points = dubinEHF3d(x1, y1, alt1, psi, x2, y2, r_min, steplength, gamma)

        # print("Number of points: ", num_path_points)

        # Check if the path is valid (non-zero points)
        if num_path_points > 0:
            inputs.append([x2, y2, psi, gamma])
            trajectories.append(path)

    return inputs, trajectories


if __name__ == "__main__":
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    input_size = 4  # x2, y2, psi, gamma
    hidden_size = 128
    num_layers = 1
    output_size = 3  # Trajectory output: x, y, alt
    model = DubinsRNN(input_size, hidden_size, output_size, num_layers=num_layers)
    model.load_state_dict(torch.load('dubins_rnn_model.pth', weights_only=True))

    model.to(device)

    model.eval()

    inputs, ground_truth_trajectories = generate_random_ground_truths()

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    predicted_trajectories = []

    # Loop through each input and get the model prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        for input_data in inputs_tensor:
            inputs = input_data.unsqueeze(0).repeat(1, 100, 1).to(device)
            prediction = model(inputs)
            predicted_trajectories.append(prediction.squeeze().tolist())  # Squeeze and convert to list

    # Convert predicted_trajectories list to a numpy array if desired
    predicted_trajectories = np.array(predicted_trajectories)

    print(predicted_trajectories.shape)
    plot_dubin_paths(ground_truth_trajectories, predicted_trajectories)


    