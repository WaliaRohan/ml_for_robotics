import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process bounding boxes and project onto an image.")
    parser.add_argument("--height", type=int, default=20, required=False, help="Height of the camera in meters.")
    parser.add_argument("--frame", type=str, default="00000", required=False, help="Frame number of the image.")
    return parser.parse_args()

def resize_with_aspect_ratio(image, target_size, plot_images=False):
    """Resize an image to the target size while preserving aspect ratio, padding with black pixels."""
    h, w = image.shape[:2]  # grab width and height of image
    target_h, target_w = target_size

    # Compute scale for resizing
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a blank canvas for the target size
    if len(image.shape) == 2:  # Single-channel (grayscale) image
        canvas = np.zeros((target_h, target_w), dtype=image.dtype)
    else:  # Multi-channel image
        canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)

    # Compute padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    # Place the resized image in the center of the canvas
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # Plot the images if requested
    if plot_images:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image if len(image.shape) == 3 else image, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(canvas if len(canvas.shape) == 3 else canvas, cmap='gray')
        axes[1].set_title("Resized Image")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    return canvas, scale, pad_x, pad_y

def getBBox(camera_file, semantic_file, bbox_file, image_file):
    # Load files
    camera = json.load(open(camera_file))
    sem = cv2.imread(semantic_file, cv2.IMREAD_UNCHANGED)
    bboxes = json.load(open(bbox_file))

    # Calculate bounding box translation and rotation
    shift = np.array([camera['x'], camera['y'], camera['z']])
    rotation = R.from_euler('yzx', [90-camera['pitch'], camera['yaw'], camera['roll']], degrees=True).as_matrix()

    # Intrinsic matrix
    K = np.array([
        [0, 960, 960],
        [960, 0, 540],
        [0, 0, 1]
    ])

    # Class ID remapping
    idmap = {(40,): 0, (100,): 1, (101,): 2, (102,): 3, (103,): 4, (104,): 5, (105,): 6, (104, 41): 7, (105, 41): 8, (41, 104): 7, (41, 105): 8}

    # Bounding boxes
    bbs = np.array([bb['corners'] for bb in bboxes]) - shift
    bbs = bbs @ rotation

    # Project bounding boxes
    pbb = bbs @ K.T

    # Check if homogeneous coordinate "w" is positive
    valid = np.any(pbb[..., -1] > 0, axis=-1)

    # Normalize bounding boxes
    pbb /= pbb[..., -1:] + 1e-5
    uls = pbb.min(axis=1)  # upper left corner
    lrs = pbb.max(axis=1)  # lower right corner

    # Load the original image
    image = cv2.imread(image_file)

    target_size = 640

    # Resize the image with aspect ratio preserved
    resized_image, scale, pad_x, pad_y = resize_with_aspect_ratio(image, (target_size, target_size))
    sem_resized, _, _, _ = resize_with_aspect_ratio(sem, (target_size, target_size))

    parent_dir = os.path.dirname(image_file)
    output_folder = os.path.join(parent_dir, "resized")

    # Save the resized image in the "resized" folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    resized_image_path = os.path.join(output_folder, os.path.basename(image_file))
    cv2.imwrite(resized_image_path, resized_image)

    # Adjust bounding boxes for the resized image
    vboxes = []
    for v, ul, lr, bb in zip(valid, uls, lrs, bboxes):
        if v:
            x0, y0 = np.round(ul).astype(int)[:2]
            x1, y1 = np.round(lr).astype(int)[:2]

            # Scale bounding box coordinates
            x0 = int(x0 * scale + pad_x)
            x1 = int(x1 * scale + pad_x)
            y0 = int(y0 * scale + pad_y)
            y1 = int(y1 * scale + pad_y)

            # Clip to image boundaries
            x0 = np.clip(x0, a_min=0, a_max=target_size)
            x1 = np.clip(x1, a_min=0, a_max=target_size)
            y0 = np.clip(y0, a_min=0, a_max=target_size)
            y1 = np.clip(y1, a_min=0, a_max=target_size)

            # Validate box area and check for matching class pixels
            if x1 > x0 and y1 > y0 and (x1 - x0) * (y1 - y0) < target_size * target_size / 2:
                roi = sem_resized[y0:y1, x0:x1]
                flag = False
                for cl in bb['class']:
                    flag = flag or np.any(roi == cl)
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

    image_folder = data_folder + "height" + str(height) + "m/rgb/"
    image_file = image_folder + frame + ".jpg"

    return getBBox(camera_file, semantic_file, bbox_file, image_file)


def plot_bbox(frame, height, resized=False):

    # Get bounding boxes
    vboxes = read_bbox(frame, height)

    # File paths
    data_folder = "syndrone_data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/"
    image_folder = data_folder + "height" + str(height) + "m/rgb/"
    
    if resized:
        image_folder = image_folder + "resized/"
    
    image_file = image_folder + frame + ".jpg"

    # Load the image
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

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

    # Draw bounding boxes
    for bbox, class_id, roi in vboxes:
        x0, y0, x1, y1 = bbox
        # Draw rectangle
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)  # Blue box with thickness 2
        # Put class
        cv2.putText(image, class_strs[str(class_id)], (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green text

    # Display the image
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def create_json(image_range, data_folder, height, resized=False):

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
        if resized:
            image_folder = image_folder + "resized/"
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
        vboxes = getBBox(camera_file, semantic_file, bbox_file, image_file)

        target_size = 640

        # Add image metadata
        output_json["images"].append({
            "id": frame_index,
            "width": target_size,
            "height": target_size,
            "file_name": frame + ".jpg"
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
                "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
            }

            output_json["annotations"].append(annotation)
            annotation_id += 1

    # Save the JSON file
    output_path = os.path.join("./annotations/annotations.json")
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)

    print(f"JSON file created at: {output_path}")
    

# Main script execution
if __name__ == "__main__":
    args = parse_arguments()  # Parse the command-line arguments


    # image_file = "syndrone_data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb/00000.jpg"
    # image = cv2.imread(image_file)
    # resize_with_aspect_ratio(image, (1080, 1080), True)

    height = args.height
    frame = args.frame

    # plot_bbox(frame, height, True)

    image_range = range(0, 10)  # Change range as needed
    data_folder = "syndrone_data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/"

    # # Create the JSON file
    create_json(image_range, data_folder, height)



    