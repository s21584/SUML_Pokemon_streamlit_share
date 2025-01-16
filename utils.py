import os
import ast
import io
import pandas as pd
from typing import List, Dict
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

# Optional: This ensures that we can open truncated images if needed.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_dataset(data_path: str) -> List[Dict[str, Image.Image]]:
    """
    Converts a CSV dataset into a list of dictionaries for model training.

    This function reads a CSV file containing at least two columns:
      - 'image': A string representation of binary image data (in bytes).
      - 'better_description': A human-generated description of the image.

    It converts each row's binary image data into an RGB PIL image and pairs it
    with the associated description. The output is a list of dictionaries
    suitable for use with Transformers or similar libraries.

    Args:
        data_path (str):
            The path to the CSV file to load.

    Returns:
        List[Dict[str, Image.Image]]:
            A list of dictionaries. Each dictionary contains:
                {
                    "image": <PIL.Image.Image object in RGB>,
                    "text": <string description from the CSV>
                }

    Raises:
        FileNotFoundError:
            If the specified CSV file cannot be found.
        ValueError:
            If the file format or column structure is not as expected.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if "image" not in df.columns or "better_description" not in df.columns:
        raise ValueError(
            "CSV file must contain 'image' and 'better_description' columns."
        )

    data = []
    for _, row in df.iterrows():
        # Convert the image field (string of bytes) to a bytes object
        image_bytes = bytes(ast.literal_eval(row["image"])["bytes"])
        rgb_image = binary_to_rgb(image_bytes)

        data.append({
            "image": rgb_image,
            "text": row["better_description"]
        })

    return data


def binary_to_rgb(binary_data: bytes) -> Image.Image:
    """
    Converts binary image data into an RGB PIL Image.

    Args:
        binary_data (bytes):
            The binary data representing an image (e.g., PNG or JPEG).

    Returns:
        Image.Image:
            A PIL Image object converted to RGB mode.

    Raises:
        OSError:
            If the provided bytes do not represent a valid image file.
    """
    image = Image.open(io.BytesIO(binary_data))
    return image.convert("RGB")


def split_data(data_path: str) -> None:
    """
    Splits a CSV dataset into training and testing subsets and saves them as separate CSV files.

    This function reads a CSV file, splits it into train/test sets using
    an approximate 90/10 ratio, and saves the resulting subsets to:
      - `data/train.csv`
      - `data/test.csv`

    Args:
        data_path (str):
            The path to the CSV file to be split.

    Returns:
        None

    Raises:
        FileNotFoundError:
            If the specified CSV file cannot be found.
        ValueError:
            If the dataset is too small or missing expected columns.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("The dataset is empty and cannot be split.")

    train_df, test_df = train_test_split(df, test_size=0.1)

    # Create output directories if they don't exist
    os.makedirs("data", exist_ok=True)

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
