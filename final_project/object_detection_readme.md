# README: Object Detection Dataset Preparation and Training

## Required Packages

| Package                | Version    |
|------------------------|------------|
| Python                | 3.11.7     |
| matplotlib            | 3.9.2      |
| numpy                 | 2.1.3      |
| onnx                  | 1.17.0     |
| onnxruntime-gpu       | 1.20.1     |
| onnxslim              | 0.1.43     |
| opencv-python         | 4.10.0.84  |
| pandas                | 2.2.3      |
| pillow                | 11.0.0     |
| pip                   | 23.2.1     |
| torch                 | 2.5.1      |
| torchvision           | 0.20.1     |
| tqdm                  | 4.67.0     |
| ultralytics           | 8.3.34     |
| nvidia-cublas-cu12    | 12.4.5.8   |
| nvidia-cuda-runtime-cu12 | 12.4.127 |
| nvidia-cudnn-cu12     | 9.1.0.70   |

---

## 1. Creating the Dataset for Training

### A. Create COCO Dataset

You can use the provided dataset or recreate the COCO dataset using the following file:

```bash
python3 process_bounding_boxes.py
```

This script should be run from the same directory containing the `syndrone_dataset` directory (see the file for more details). Upon successful execution, you will see:

```
JSON file created at: annotations.json
```

This file contains all 2D bounding box annotations from Syndrone in COCO format. The script also resizes all images (Town 1, height 20m by default) to `640x640` and saves them to:

```
<original_parent_directory_of_rgb_folder>/resized
```

### B. Convert COCO Dataset to YOLO

Navigate to the `JSON2YOLO` subdirectory and run:

```bash
python3 general_json2yolo.py
```

You should see output similar to:

```
Annotations <path_to_annotations_json_parent_directory>/annotations.json: 100%|█| 2967/2967 [00:00<00:00, 3352.48it]
```

This will generate the YOLO annotations and corresponding images in the `dataset` folder within the current directory.

### C. Shuffle Images and Annotations

Perform a randomized train/test split by running:

```bash
python3 create_train_test_split.py
```

Expected output:

```
Copying train files: 100%|████████████████████████████████████████████████████████████████████████████| 2076/2076 [00:00<00:00, 4454.66it/s]
Copying val files: 100%|████████████████████████████████████████████████████████████████████████████████| 891/891 [00:00<00:00, 4383.88it/s]
Files have been successfully copied.
```

Finally, copy the `dataset` folder in `JSON2YOLO` to the same directory as the training script. Place the `dataset.yaml` file in it. The dataset is now ready for training using a YOLO model.

---

## 2. Training

Ensure the dataset is in the same directory as the training script, then run:

```bash
python3 train.py
```

The training results will be saved under the `runs` directory in the parent directory of the current folder.

### Optional: View Training and Validation Losses

Run the following script to visualize training and validation losses:

```bash
python3 plot_losses.py
```

---

### Optional: Count and view annotations for each image

Run the following script to see how many annotations arae there in each image: 

```bash
python3 count_annotations.py
```

## 3. Predictions

10 predictions have already been saved in the "generated predictions" folder. To generate predictions from scratch, use:

```bash
python3 generate_predictions.py
```

Ensure the original Syndrone dataset is saved in the same directory. Refer to `processing_bounding_boxes.py` for details.

