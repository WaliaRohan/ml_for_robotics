import os

import cv2
import numpy as np
from PIL import Image

from process_bounding_boxes import plot_bbox
from ultralytics import YOLO

target_dir = "/home/speedracer1702/Projects/academic/ml_for_robotics/final_project/syndrone_data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb/resized"
image_indexes = [
    "00163", "02603", "00249", "00017", "01017", 
    "02090", "01324", "02259", "00714", "00816"
]

# Example: Define a pre-trained model (replace with your own model)
model = YOLO("model13_best.pt")
image_paths = [os.path.join(target_dir, f"{index}.jpg") for index in image_indexes]

for path in image_paths:
    print(path)

results = model(image_paths)
# results=[]

# Get current working directory
current_dir = os.getcwd()

# Define the predictions directory path
predictions_dir = os.path.join(os.getcwd(), "predictions")

# Create the directory if it doesn't exist
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

# Process results list
for image_index, image_path, result in zip(image_indexes, image_paths, results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # result.save(filename=".jpg")  # save to disk

    # Extract the original index from the image filename
    original_index = os.path.basename(image_path).split('.')[0]  # Extract '00163' from '/path/to/00163.jpg'
    
    # Save the result as a JPG with the original index in the current directory
    output_path = os.path.join(current_dir, "predictions", f"{original_index}_predicted.jpg")
    result.save(filename=output_path)  # Save result using Ultralytics' `save()` method

    print(f"Saved result for image {original_index} as {output_path}")

    plt = plot_bbox(image_index, 20, target_class_id=2, resized=True)

    # Save the plot
    plt.savefig(os.path.join(current_dir, "predictions", f"{original_index}_gt.jpg"), dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()
        
def combine_images_with_text(left_image_path, right_image_path, output_path, image_index):
    # Load the images
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    # Ensure images are 640x640
    left_image = cv2.resize(left_image, (640, 640))
    right_image = cv2.resize(right_image, (640, 640))

    # Create a blank canvas (height: 700 to include text, width: 1280 for two images)
    canvas_height = 700  # Reduced extra space for text
    canvas_width = 1280  # Two 640px wide images side by side
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place the images on the canvas
    canvas[30:670, :640] = left_image  # Left image
    canvas[30:670, 640:] = right_image  # Right image

    # Add top text for image index (closer to the top of the images)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)  # White text
    top_text_y = 25  # Closer to the top
    top_text = f"Image Index: {image_index}"
    text_size = cv2.getTextSize(top_text, font, font_scale, thickness)[0]
    text_x = (canvas_width - text_size[0]) // 2  # Center the text
    cv2.putText(canvas, top_text, (text_x, top_text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # Add bottom text closer to the images
    bottom_text_y = 690  # Reduced gap to bring text closer to images

    # Add "Ground truth" below the left image
    cv2.putText(canvas, "Ground truth", (160, bottom_text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # Add "Prediction" below the right image
    cv2.putText(canvas, "Prediction", (800, bottom_text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # Save the combined image
    cv2.imwrite(output_path, canvas)
    print(f"Combined image saved at {output_path}")

for image_index in image_indexes:
    gt_path = os.path.join(predictions_dir, f"{image_index}_gt.jpg")
    pred_path = os.path.join(predictions_dir, f"{image_index}_predicted.jpg")
    output_combined_path = os.path.join(predictions_dir, f"{image_index}_combined.jpg")

    combine_images_with_text(gt_path, pred_path, output_combined_path, image_index)