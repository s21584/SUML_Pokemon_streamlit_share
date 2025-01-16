import os
from typing import Dict, Any, List

from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    Trainer,
    TrainingArguments,
    BlipForConditionalGeneration,
    BlipProcessor,
    BertTokenizer, BatchEncoding,
)

from utils import process_dataset
from model_api import load_model

load_dotenv()


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME: str = "Salesforce/blip-image-captioning-large"
TRAIN_DATA_PATH: str = "data/train.csv"
TEST_DATA_PATH: str = "data/test.csv"
CHECKPOINT_DIR: str = "model"
REPO_NAME: str = "RidzIn/Pokemon-Describer"
HF_TOKEN: str = os.getenv("HF_TOKEN")

TRAINING_ARGS: Dict[str, Any] = {
    "output_dir": CHECKPOINT_DIR,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "save_steps": 500,
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "save_total_limit": 2,
}


# -----------------------------------------------------------------------------
# Data Preprocessing
# -----------------------------------------------------------------------------
def preprocess_function(
        examples: Dict[str, List[Any]],
        processor: BlipProcessor
) -> BatchEncoding:
    """
    Tokenizes and processes the images and text using the provided BlipProcessor.

    Args:
        examples: Dictionary with keys 'image' and 'text'.
        processor: A BlipProcessor object for tokenizing images and text.

    Returns:
        A dictionary containing input_ids, attention_mask, and labels,
        which align with the tokenized inputs.
    """
    inputs = processor(
        images=examples["image"],
        text=examples["text"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    inputs["labels"] = inputs["input_ids"]
    return inputs


# -----------------------------------------------------------------------------
# Model Training
# -----------------------------------------------------------------------------
def train_model(
        train_dataset: Dataset,
        eval_dataset: Dataset,
        model: BlipForConditionalGeneration,
        training_args: Dict[str, Any]
) -> None:
    """
    Trains the model using the specified training and evaluation datasets.

    Args:
        train_dataset: The preprocessed training dataset.
        eval_dataset: The preprocessed evaluation/test dataset.
        model: The BlipForConditionalGeneration model to train.
        training_args: Dictionary of training arguments.
    """
    args = TrainingArguments(**training_args)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()


# -----------------------------------------------------------------------------
# Model Pushing to Hugging Face
# -----------------------------------------------------------------------------
def push_model_to_hub(
        checkpoint_dir: str,
        model_name: str,
        repo_name: str,
        token: str
) -> None:
    """
    Pushes the latest checkpoint of the model to the Hugging Face Hub.

    Args:
        checkpoint_dir: The directory where checkpoints are saved.
        model_name: The base model name used to load tokenizer and processor.
        repo_name: The Hugging Face repo name (e.g. 'username/repo').
        token: Your Hugging Face access token.
    """
    # Find the latest checkpoint
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
    if not checkpoints:
        msg = "No checkpoint found in the model directory."
        print(msg)
        raise FileNotFoundError(msg)

    # Use the latest checkpoint
    latest_checkpoint = sorted(checkpoints)[-1]
    model_path = os.path.join(checkpoint_dir, latest_checkpoint)

    # Load model, tokenizer, and processor from the latest checkpoint
    model = BlipForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    processor = BlipProcessor.from_pretrained(model_name)

    # Push model, tokenizer, and processor to Hugging Face Hub
    model.push_to_hub(repo_name, token=token)
    tokenizer.push_to_hub(repo_name, token=token)
    processor.push_to_hub(repo_name, token=token)
    print(f"Model and tokenizer pushed to Hugging Face Hub under {repo_name}")


# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main execution function that orchestrates:
    1. Loading the model and processor.
    2. Processing data.
    3. Training the model.
    4. Pushing the trained model to the Hugging Face Hub.
    """
    # 1. Load model and processor
    processor, model = load_model(MODEL_NAME)

    # 2. Process the training dataset
    raw_train_data: List[Dict[str, Any]] = process_dataset(TRAIN_DATA_PATH)
    train_dataset: Dataset = Dataset.from_list(raw_train_data)
    processed_train_dataset: Dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, processor),
        batched=True
    )

    # 3. Process the testing dataset
    raw_test_data: List[Dict[str, Any]] = process_dataset(TEST_DATA_PATH)
    test_dataset: Dataset = Dataset.from_list(raw_test_data)
    processed_test_dataset: Dataset = test_dataset.map(
        lambda examples: preprocess_function(examples, processor),
        batched=True
    )

    # 4. Train the model
    train_model(
        train_dataset=processed_train_dataset,
        eval_dataset=processed_test_dataset,
        model=model,
        training_args=TRAINING_ARGS
    )

    # 5. Push the model to Hugging Face Hub
    push_model_to_hub(
        checkpoint_dir=CHECKPOINT_DIR,
        model_name=MODEL_NAME,
        repo_name=REPO_NAME,
        token=HF_TOKEN
    )


if __name__ == "__main__":
    main()
