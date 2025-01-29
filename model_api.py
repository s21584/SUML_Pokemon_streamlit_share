import os
import torch
from PIL import Image
from datasets import Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Dict, Tuple, Optional, List

from utils import process_dataset


def load_model(model_name: str) -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """
    Loads a BLIP model and processor from the Hugging Face Hub.

    Args:
        model_name (str): The name of the model on Hugging Face Hub.
                          e.g., "Salesforce/blip-image-captioning-large", "RidzIn/Pokemon-Describer"

    Returns:
        Tuple[BlipProcessor, BlipForConditionalGeneration]:
            - processor: The BLIP processor for tokenizing images and text.
            - model: The BLIP model (for conditional generation).
    """
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model


def invoke_model(
    model_name: str,
    image_path: str,
    caption: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    Generates a caption for a single image using the specified model.

    Args:
        model_name (str): The name of the Hugging Face model to use.
        image_path (str): The local file path to the image (png or jpeg).
        caption (Optional[str]): A human-written caption or description for reference.

    Returns:
        Dict[str, Optional[str]]:
            {
                "generated_description": [The generated caption],
                "original": [The original provided caption, if any]
            }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load processor and model
    processor, model = load_model(model_name)

    # Open and convert the image to RGB
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=128)
        generated_text: str = processor.decode(generated_ids[0], skip_special_tokens=True)

    return {
        "generated_description": generated_text,
        "original": caption
    }


def invoke_batch(
    model_name: str,
    data_path: str
) -> Dict[str, List[str]]:
    """
    Generates captions for a batch of images stored in a dataset CSV or JSON file.

    Args:
        model_name (str): The name of the Hugging Face model to use.
        data_path (str): The path to the dataset file containing 'image' and 'text' fields.

    Returns:
        Dict[str, List[str]]:
            {
                "generated_descriptions": [List of generated descriptions],
                "originals": [List of original descriptions from the dataset]
            }
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load processor and model
    processor, model = load_model(model_name)

    # Process dataset (the process_dataset function should return a list of dicts: [{"image": ..., "text": ...}, ...])
    data = process_dataset(data_path)
    dataset = Dataset.from_list(data)

    # Preprocess all images
    inputs = processor(images=dataset["image"], return_tensors="pt")

    result = {
        "generated_descriptions": [],
        "originals": []
    }

    # Generate captions for each image
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=128)
        for i in range(generated_ids.shape[0]):
            generated_text: str = processor.decode(generated_ids[i], skip_special_tokens=True)
            result["generated_descriptions"].append(generated_text)
            result["originals"].append(dataset["text"][i])

    return result
