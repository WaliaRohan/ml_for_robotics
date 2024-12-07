import os

import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
csv_path = "../runs/detect/train/results.csv"
data = pd.read_csv(csv_path)

# Extract the directory of the CSV file
output_dir = os.path.dirname(csv_path)

# Extract relevant columns
epoch = data["epoch"]
train_box_loss = data["train/box_loss"]
val_box_loss = data["val/box_loss"]
train_cls_loss = data["train/cls_loss"]
val_cls_loss = data["val/cls_loss"]
train_dfl_loss = data["train/dfl_loss"]
val_dfl_loss = data["val/dfl_loss"]

# Plot 1: Box Loss
plt.figure()
plt.plot(epoch, train_box_loss, label="Train Box Loss", color="blue")
plt.plot(epoch, val_box_loss, label="Validation Box Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Box Loss")
plt.legend()
plt.grid(True)
box_loss_path = os.path.join(output_dir, "box_loss.png")
plt.savefig(box_loss_path)
plt.show()

# Plot 2: Classification Loss
plt.figure()
plt.plot(epoch, train_cls_loss, label="Train Classification Loss", color="green")
plt.plot(epoch, val_cls_loss, label="Validation Classification Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Classification Loss")
plt.legend()
plt.grid(True)
cls_loss_path = os.path.join(output_dir, "classification_loss.png")
plt.savefig(cls_loss_path)
plt.show()

# Plot 3: Distribution Focal Loss
plt.figure()
plt.plot(epoch, train_dfl_loss, label="Train DFL Loss", color="purple")
plt.plot(epoch, val_dfl_loss, label="Validation DFL Loss", color="brown")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Distribution Focal Loss")
plt.legend()
plt.grid(True)
dfl_loss_path = os.path.join(output_dir, "distribution_focal_loss.png")
plt.savefig(dfl_loss_path)
plt.show()

print("Plots saved as:")
print(f"1. {box_loss_path}")
print(f"2. {cls_loss_path}")
print(f"3. {dfl_loss_path}")
