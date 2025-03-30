import logging
import os

import pandas as pd
from datasets import ClassLabel, Dataset, Features, Image, Value
from huggingface_hub import login

from settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dataset_creation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = (
    "/Users/vesaalexandru/Workspaces/cube/fashion-recommender-system/fashion_data"
)
IMAGE_DIR = os.path.join(DATA_DIR, "images_compressed")
CSV_PATH = os.path.join(DATA_DIR, "images.csv")

# Check if directories exist
logger.info(f"Checking image directory: {IMAGE_DIR}")
if not os.path.exists(IMAGE_DIR):
    logger.error(f"Image directory not found: {IMAGE_DIR}")
    exit(1)

# Read the CSV file
logger.info(f"Reading CSV file: {CSV_PATH}")
try:
    # Read with header (CSV has headers as shown in the example)
    df = pd.read_csv(CSV_PATH)
    logger.info(f"CSV loaded with {len(df)} rows")
    logger.info(f"CSV columns: {df.columns.tolist()}")
    logger.info(f"First row as example: {df.iloc[0].to_dict()}")
except Exception as e:
    logger.error(f"Error reading CSV: {e}")
    exit(1)

# Get unique categories for label encoding
categories = sorted(df["label"].unique().tolist())  # Use "label" instead of "category"
logger.info(f"Found {len(categories)} unique categories")

# Create image-label pairs
logger.info("Processing images...")
image_paths = []
labels = []
valid_indices = []

for idx, row in df.iterrows():
    if row["kids"] == False:  # Use "kids" column instead of "is_deleted"
        image_path = os.path.join(
            IMAGE_DIR, f"{row['image']}.jpg"
        )  # Use "image" column
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(row["label"])  # Use "label" column
            valid_indices.append(idx)
        else:
            if len(image_paths) == 0 and idx < 5:  # Only log a few at the beginning
                logger.warning(f"Image not found: {image_path}")

logger.info(f"Found {len(valid_indices)} valid images out of {len(df)} entries")

# Check if we have any valid images
if len(valid_indices) == 0:
    logger.error("No valid images found. Please check image paths and CSV data.")
    exit(1)

# Create the dataset
logger.info("Creating dataset...")
dataset_dict = {
    "image": image_paths,
    "label": labels,
    "image_id": df.iloc[valid_indices]["image"].tolist(),
    "user_id": df.iloc[valid_indices]["sender_id"].tolist(),
}

features = Features(
    {
        "image": Image(),
        "label": ClassLabel(names=categories),
        "image_id": Value("string"),
        "user_id": Value("int64"),
    }
)

dataset = Dataset.from_dict(dataset_dict, features=features)

# Save the dataset
logger.info("Saving dataset...")
dataset.save_to_disk("fashion_dataset")
logger.info(f"Dataset saved with {len(dataset)} samples")

# Push to Hub with login
logger.info("Pushing dataset to Hugging Face Hub...")
try:
    login(token=settings.data.HF_AUTH_TOKEN)

    dataset.push_to_hub("valex95/fashion-dataset")
    logger.info("Dataset successfully pushed to Hub")
except Exception as e:
    logger.error(f"Failed to push to Hub: {e}")
    logger.info("Dataset was saved locally at 'fashion_dataset'")
