import os
import random
import shutil

from tqdm import \
    tqdm  # Install tqdm if not already installed: pip install tqdm

# Paths
label_folder = os.path.join(os.getcwd(), "dataset/labels/")
images_source_folder = "/home/speedracer1702/Projects/academic/ml_for_robotics/final_project/syndrone_data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb/resized"
images_dest_folder = os.path.join(os.getcwd(), "dataset/images/")

# Output directories
train_text_folder = os.path.join(os.path.join(os.getcwd(), "dataset/labels/"), "train")
val_text_folder = os.path.join(os.path.join(os.getcwd(), "dataset/labels/"), "val")
train_image_folder = os.path.join(images_dest_folder, "train")
val_image_folder = os.path.join(images_dest_folder, "val")

# Create folders if they don't exist
os.makedirs(train_text_folder, exist_ok=True)
os.makedirs(val_text_folder, exist_ok=True)
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)

# Read filenames without extensions
text_files = [os.path.splitext(f)[0] for f in os.listdir(label_folder) if f.endswith(".txt")]

# Randomize the list and split
random.seed(17)
random.shuffle(text_files)
train_ratio = 0.7
split_idx = int(train_ratio * len(text_files))
train_files = text_files[:split_idx]
val_files = text_files[split_idx:]

# Function to copy files
def copy_files(files, split):
    for file in tqdm(files, desc=f"Copying {split} files"):
        # Copy label files
        label_src = os.path.join(label_folder, file + ".txt")
        label_dest = os.path.join(label_folder, split, file + ".txt")
        shutil.copy(label_src, label_dest)

        # Copy image files with .jpg extension
        image_src = os.path.join(images_source_folder, file + ".jpg")
        image_dest = os.path.join(images_dest_folder, split, file + ".jpg")
        if os.path.exists(image_src):
            shutil.copy(image_src, image_dest)
        else:
            print(f"Warning: Image not found for {file}. Skipping.")

# Copy files to train and val
copy_files(train_files, "train")
copy_files(val_files, "val")

print("Files have been successfully copied.")