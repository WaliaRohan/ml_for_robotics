import json
import math


def getCameraProjectionArea(camera_file, height):
    """
    Calculate the x and y coordinate range of the rectangular area visible to a downward-facing camera.
    
    Parameters:
        camera_file (str): Path to the JSON file containing the camera pose.

    Returns:
        tuple: A tuple containing two tuples:
            - x_range (float, float): The minimum and maximum x-coordinates.
            - y_range (float, float): The minimum and maximum y-coordinates.
    """
    # Load the camera pose from the JSON file
    with open(camera_file, 'r') as f:
        camera_pose = json.load(f)

    # Vertical and horizontal FOVs (in degrees)
    if height == 80:
        vertical_fov = 90  # Vertical FOV in degrees
        horizontal_fov = 121.2  # Horizontal FOV in degrees (calculated from resolution and aspect ratio)
    elif height == 50:
        vertical_fov = 60
        horizontal_fov = 110
    elif height == 20:
        vertical_fov = 30
        horizontal_fov = 60

    # Convert FOVs to radians
    vertical_fov_rad = math.radians(vertical_fov / 2)
    horizontal_fov_rad = math.radians(horizontal_fov / 2)

    # Camera properties
    height_above_ground = camera_pose["z"]  # Height of the camera above ground in meters
    camera_x = camera_pose["x"]  # Camera x position in meters
    camera_y = camera_pose["y"]  # Camera y position in meters

    # Calculate projected area dimensions
    projected_width = 2 * height_above_ground * math.tan(horizontal_fov_rad)
    projected_height = 2 * height_above_ground * math.tan(vertical_fov_rad)

    # Calculate coordinate ranges
    x_min = camera_x - projected_width / 2
    x_max = camera_x + projected_width / 2
    y_min = camera_y - projected_height / 2
    y_max = camera_y + projected_height / 2

    return (x_min, x_max), (y_min, y_max)


if __name__ == "__main__":
    # Example usage of the function
    data_folder = "syndrone_data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/"

    camera_file = data_folder + "height80m/camera/" + "00000.json"

    x_range, y_range = getCameraProjectionArea(camera_file, 80)
    print("Coordinate Ranges:")
    print(f"x-range: {x_range}")
    print(f"y-range: {y_range}")
