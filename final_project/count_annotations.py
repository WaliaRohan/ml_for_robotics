import csv
import json
from collections import defaultdict

# Define the category mapping
category_mapping = {
    1: "Person",
    2: "Bicycle",
    3: "Car",
    4: "Motorcycle",
    6: "Bus",
    7: "Train",
    8: "Truck"
}

# File paths
input_json = "annotations.json"
output_csv = "image_category_counts.csv"

# Load the COCO JSON file
with open(input_json, "r") as file:
    data = json.load(file)

# Create a dictionary to store counts for each image
image_category_counts = defaultdict(lambda: {cat: 0 for cat in category_mapping.values()})

# Iterate through annotations and update counts
for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]
    if category_id in category_mapping:
        category_name = category_mapping[category_id]
        image_category_counts[image_id][category_name] += 1

# Write results to a CSV file
with open(output_csv, "w", newline="") as file:
    writer = csv.writer(file)
    # Write header
    header = ["Image #"] + list(category_mapping.values())
    writer.writerow(header)

    # Write counts for each image
    for image_id, counts in image_category_counts.items():
        row = [image_id] + [counts[cat] for cat in category_mapping.values()]
        writer.writerow(row)

print(f"Image category counts have been saved to {output_csv}")
