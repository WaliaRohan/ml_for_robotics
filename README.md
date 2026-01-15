# ML for Robotics

This repository contains a collection of focused machine learning and control projects applied to robotics. Each subfolder represents an independent project and includes its own README with detailed explanations, methods, and results.

The work emphasizes learning-based control, trajectory generation, and robotics-oriented ML pipelines, with a focus on physical consistency and practical deployment.

## Highlighted Projects

### Neural Network Control Allocation

A learning-based control allocation framework for dynamic positioning of autonomous ships. Implements an encoder–decoder neural network that maps desired force/moment commands to thruster inputs while respecting actuator limits and minimizing power consumption.

### LSTM-Based Trajectory Generation

An LSTM-based model for generating dynamically feasible 3D Dubins trajectories using only initial pose, goal position, and climb angle. Trained on a large synthetic dataset of variable-length trajectories to produce smooth, constraint-aware paths.

## Repository Structure

* Each folder contains a dedicated README with setup instructions.

## Technologies

Python, PyTorch, ROS / ROS 2, MATLAB, Linux, Docker, Git

## Author

Rohan Walia
GitHub: [https://github.com/WaliaRohan](https://github.com/WaliaRohan)
