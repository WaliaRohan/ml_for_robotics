import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from camera_projection import getCameraProjectionArea  # Import the function

# File paths
data_folder = "syndrone_data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/"

height = 80

image_folder = data_folder + "height" + str(height) + "m/rgb/"
bbox_folder = data_folder + "bboxes/"
camera_file = data_folder + "height" + str(height) + "m/camera/" + "00000.json"

image_file = image_folder + "00000.jpg"
bbox_file = bbox_folder + "00000.json"

# Function to transform points to the camera frame
def transform_to_camera_frame(corners, camera_pose):
    # Extract camera translation and rotation
    cam_pos = np.array([camera_pose["x"], camera_pose["y"], camera_pose["z"]])
    cam_rot = R.from_euler('xyz', [camera_pose["pitch"], camera_pose["yaw"], camera_pose["roll"]], degrees=True).as_matrix()
    
    # Transform corners to camera frame
    corners = np.array(corners)  # Convert to numpy array
    corners_camera_frame = (corners - cam_pos) @ cam_rot.T  # Apply translation and rotation
    return corners_camera_frame

# Read the image
image = cv2.imread(image_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB

plt.figure(figsize=(10, 10))
# plt.imshow(image)
plt.title("Image with Bounding Boxes in Camera Frame")

# Read the bounding box JSON
with open(bbox_file, 'r') as f:
    bounding_boxes = json.load(f)

def printCoordinates(x_range, y_range):
    print(f"x-range: {x_range}")
    print(f"y-range: {y_range}")    

# Read the camera JSON
x_range, y_range = getCameraProjectionArea(camera_file, height)  # Get camera projection area
print("Camera Projection Area Coordinates:")


num_valid_bounding_boxes = 0

for bbox in bounding_boxes:
    if 100 in bbox["class"]:  # Filter for class 100 - cars only
        corners_world = bbox["corners"]
        
        corners_camera = corners_world

        # Check if all corners are within the projection area
        all_in_projection_area = all(
            (x_range[0] <= corner[0] <= x_range[1]) and (y_range[0] <= corner[1] <= y_range[1])
            for corner in corners_world
        )
        if not all_in_projection_area:
            continue  # Skip bounding box if any corner is outside the camera's projection area

        corners_camera = transform_to_camera_frame(corners_world, json.load(open(camera_file, 'r')))  # Transform corners

        # Extract 2D coordinates (x, y) from the transformed corners
        corners_2d = [corners_camera[i][:2] for i in [0, 2, 4, 6]]  # Use corners 1, 3, 5, and 7 since z dimension doesn't matter
        corners_2d.append(corners_2d[0])  # Close the loop for the bounding box
        x_coords, y_coords = zip(*corners_2d)
        plt.plot(x_coords, y_coords, marker='o', label=f"Class: {bbox['class'][0]}, ID: {bbox['id']}")
        num_valid_bounding_boxes += 1
        print(corners_2d)

plt.legend()
plt.show()

plt.savefig("bbox_camera_frame_all_corners_filtered_class_100.png", dpi=300, bbox_inches='tight')  # Save the plot
plt.close()  # Close the figure to avoid displaying it in interactive environments

print("Number of valid bounding boxes: ", num_valid_bounding_boxes)
