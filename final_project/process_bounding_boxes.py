import argparse
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process bounding boxes and project onto an image.")
    parser.add_argument("--height", type=int, default=80, required=False, help="Height of the camera in meters.")
    parser.add_argument("--frame", type=str, default="00000", required=False, help="Frame number of the image.")
    return parser.parse_args()

def getBBox(camera_file, semantic_file, bbox_file):

    camera = json.load(open(camera_file))
    sem = cv2.imread(semantic_file, cv2.IMREAD_UNCHANGED)
    bboxes = json.load(open(bbox_file))

    # Calculate bounding box translation and rotation
    shift = np.array([camera['x'], camera['y'], camera['z']])
    rotation = R.from_euler('yzx', [90-camera['pitch'], camera['yaw'], camera['roll']], degrees=True).as_matrix()

    # This is likely the intrinsic matrix - allows projecting 3D points onto the 2D image plane
    K = np.array([
    [0, 960, 960],
    [960, 0, 540],
    [0, 0, 1]])

    # This remaps class numbers
    idmap = {(40,):0, (100,):1, (101,):2, (102,):3, (103,):4, (104,):5, (105,):6, (104, 41):7, (105, 41):8, (41, 104):7, (41, 105):8}

    # Bounding boxes
    bbs = np.array([bb['corners'] for bb in bboxes]) - shift
    bbs = bbs @ rotation

    # Projected bounding boxes - obtained when the 3D bounding boxes are multiplied by the camera's intrinsic matrix
    pbb = bbs @ K.T

    # After a point is projected into 2D homogeneous coordinates, its third value (often called w) should be positive
    # to indicate that the point is in front of the camera (visible). If w ≤ 0, the point is behind the camera or invalid for projection.
    
    # Check if homogeneous coordinate "w" (third column of pbb) is positive. 
    valid = np.any(pbb[...,-1] > 0, axis=-1)

    # Normalize bounding boxes
    pbb /= pbb[...,-1:] + 1e-5
    uls = pbb.min(axis=1) # upper left corner
    lrs = pbb.max(axis=1) # lower right corner

    vboxes = [] # list of acceptable 2D bounding boxes
    for v, ul, lr, bb in zip(valid, uls, lrs, bboxes):
        # If point is valid (in front of camera)
        if v:
            # Round to integers
            x0, y0 = np.round(ul).astype(int)[:2]
            x1, y1 = np.round(lr).astype(int)[:2]
            # Ensure coordinates are within image boundaries
            x0 = np.clip(x0, a_min=0, a_max=1920)
            x1 = np.clip(x1, a_min=0, a_max=1920)
            y0 = np.clip(y0, a_min=0, a_max=1080)
            y1 = np.clip(y1, a_min=0, a_max=1080)

            # The lower-right corner (x1, y1) must be strictly greater than the upper-left corner (x0, y0), ensuring the box has positive area.
            # The area of the bounding box (width x height) must be less than half the image size (1920*1080/2). This removes overly large bounding boxes.
            if x1 > x0 and y1 > y0 and (x1-x0)*(y1-y0) < 1920*1080/2:
                roi = sem[y0:y1, x0:x1] # extract segmentation mask
                flag = False
                for cl in bb['class']:
                    flag = flag or np.any(roi == cl) # check if any pixel in roi matches classes present in bb
                if flag:
                    vboxes.append(([x0, y0, x1, y1], idmap[tuple(bb['class'])], roi))

    return vboxes


def read_bbox(frame, height):

    # File paths
    data_folder = "syndrone_data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/"

    bbox_folder = data_folder + "bboxes/"
    camera_file = data_folder + "height" + str(height) + "m/camera/" + frame + ".json"

    bbox_file = bbox_folder + frame + ".json"
    semantic_file = data_folder + "height" + str(height) + "m/semantic/" + frame + ".png"

    return getBBox(camera_file, semantic_file, bbox_file)


def plot_bbox(frame, height):
    
    # File paths
    data_folder = "syndrone_data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/"
    image_folder = data_folder + "height" + str(height) + "m/rgb/"
    image_file = image_folder + frame + ".jpg"
    
    # Load the image
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    vboxes = read_bbox(frame, height)

    # Draw bounding boxes
    for bbox, class_id, roi in vboxes:
        x0, y0, x1, y1 = bbox
        # Draw rectangle
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)  # Blue box with thickness 2
        # Put class ID text
        cv2.putText(image, str(class_id), (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green text

    # Display the image
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def create_json(image_range, data_folder, height):

    # References for what JSON should look like: 
    # https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html 
    # https://cocodataset.org/#format-data 

    class_strs = {
    '0': 'Persons', # Original class 40
    '1': 'Cars', # Original class 100
    '2': 'Trucks', # Original class 101
    '3': 'Busses', # Original class 102
    '4': 'Trains', # Original class 103
    '5': 'Motorcycles', # Original class 104 
    '6': 'Bicycles', # Original class 105
    '7': 'Riders/Motorcycles', # Original class 41/104 
    '8': 'Riders/Bicycles' # Original class 41/105
    }

    output_json = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": int(key), "name": value} for key, value in class_strs.items()
        ]
    }
    
    annotation_id = 1  # Unique ID for each annotation
    for frame_index in image_range:
        frame = str(frame_index).zfill(5)

        # File paths
        image_folder = os.path.join(data_folder, f"height{height}m/rgb/")
        bbox_folder = os.path.join(data_folder, "bboxes/")
        camera_file = os.path.join(data_folder, f"height{height}m/camera/{frame}.json")
        image_file = os.path.join(image_folder, f"{frame}.jpg")
        bbox_file = os.path.join(bbox_folder, f"{frame}.json")
        semantic_file = os.path.join(data_folder, f"height{height}m/semantic/{frame}.png")

        # Check if the required files exist
        if not os.path.exists(image_file) or not os.path.exists(camera_file) or not os.path.exists(bbox_file) or not os.path.exists(semantic_file):
            print(f"Skipping frame {frame}: missing files.")
            continue

        # Process the image and bounding boxes
        vboxes = getBBox(camera_file, semantic_file, bbox_file)

        # Add image metadata
        output_json["images"].append({
            "id": frame_index,
            "width": 1920,
            "height": 1080,
            "file_name": image_file
        })

        # Add annotations
        for vbox in vboxes:
            bbox, category_id, roi = vbox
            x0, y0, x1, y1 = bbox
            area = (x1 - x0) * (y1 - y0)

            annotation = {
                "id": annotation_id,
                "category_id": category_id,
                "iscrowd": 0,
                # "segmentation": roi.tolist(),  # Add segmentation data if available
                "image_id": frame_index,
                "bbox": str([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
            }

            output_json["annotations"].append(annotation)
            annotation_id += 1

    # Save the JSON file
    output_path = os.path.join("annotations.json")
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)

    print(f"JSON file created at: {output_path}")
    

# Main script execution
if __name__ == "__main__":
    args = parse_arguments()  # Parse the command-line arguments

    height = args.height
    frame = args.frame

    # plot_bbox(frame, height)

    image_range = range(0, 10)  # Change range as needed
    data_folder = "syndrone_data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/"

    # Create the JSON file
    create_json(image_range, data_folder, height)

    