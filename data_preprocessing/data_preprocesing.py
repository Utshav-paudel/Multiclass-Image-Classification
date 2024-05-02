import json
import os
from sklearn.model_selection import train_test_split
import shutil
import subprocess

def prepare_dataset(json_file, download_dir, image_size=(299, 299), test_size=0.2):
    # Parse the JSON file
    with open(json_file) as f:
        data = [json.loads(line) for line in f]

    # Create directories to store downloaded images
    os.makedirs(download_dir, exist_ok=True)

    # Download images and organize them into classes
    images = []
    labels = []
    for item in data:
        image_url = item["imageGcsUri"]
        class_annotations = item["classificationAnnotations"]
        if class_annotations:
            class_name = class_annotations[0]["displayName"]
            image_name = os.path.basename(image_url)
            images.append(image_name)
            labels.append(class_name)
            class_dir = os.path.join(download_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            subprocess.run(["gsutil", "cp", image_url, class_dir])

    # Split data into train and test sets
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=test_size, random_state=42)

    # Move images to train and test directories
    train_dir = os.path.join(download_dir, "train")
    test_dir = os.path.join(download_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for image, label in zip(images_train, labels_train):
        src = os.path.join(download_dir, label, image)
        dst = os.path.join(train_dir, label, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)

    for image, label in zip(images_test, labels_test):
        src = os.path.join(download_dir, label, image)
        dst = os.path.join(test_dir, label, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)

    print("Dataset prepared successfully!")

# Example usage
json_file = "data.jsonl"  # Replace with the path to your JSON file
download_dir = "datasetss"  # Directory to store downloaded images
prepare_dataset(json_file, download_dir)
