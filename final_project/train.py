import os

from ultralytics import YOLO

# Get current working directory
cwd = os.getcwd()

# Add "dataset/dataset.yaml" to the path
dataset_path = os.path.join(cwd, "dataset", "dataset.yaml")
# print(dataset_path)

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11m.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data=dataset_path,
                      epochs=100,
                      classes=[2],
                      weight_decay=0.00025,
                      lr0=0.0025,
                    #   dropout=0.25,
                    #   patience=5,
                    #   box=4.5,
                    #   cls=2.0,
                    #   dfl=0.1,
                      seed=17)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")

