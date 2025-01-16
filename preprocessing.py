import os
import ast
import base64

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Union

from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
MODEL_NAME: str = "gpt-4o-mini"
SYSTEM_MESSAGE: str = """
You will be provided with a Pokémon image.
Your task is to create a short description of the Pokémon image.

Example of your output:
This character resembles a fantasy creature that looks like a combination of a mineral 
and a living being. He has blue eyes, a large white “mustache”, a pointed crystal on his head, 
and a stone body with flecks of ice crystals.
"""


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def make_better_description(image_bytes: bytes) -> str:
    """
    Generates a short description for a given Pokémon image.

    Args:
        image_bytes: The raw bytes representing an image.

    Returns:
        A string containing a short description of the Pokémon image.
    """
    # Create a ChatOpenAI instance
    model = ChatOpenAI(model=MODEL_NAME)

    # Convert image bytes to base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Create the content with the base64-encoded image
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        }
    ]

    # Create the human message
    message = HumanMessage(content=content)

    # Build the prompt
    assistant_prompt = ChatPromptTemplate.from_messages([
        message,
        ("system", SYSTEM_MESSAGE)
    ])

    # Build the chain
    assistant_chain = assistant_prompt | model | StrOutputParser()

    # Invoke the chain and return the response
    return assistant_chain.invoke({})


def add_better_description_to_dataset(data_path: str) -> None:
    """
    Reads a CSV file containing Pokémon data, generates a short description for each image,
    and appends the description to the CSV file.

    Args:
        data_path (str): The path to the CSV file to read and update.

    Raises:
        FileNotFoundError: If the CSV file at the given path does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    df = pd.read_csv(data_path)
    new_descriptions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding descriptions"):
        # Extract the image bytes from the row
        image_data = ast.literal_eval(row["image"])
        image_bytes = bytes(image_data["bytes"])

        # Generate a better description
        description = make_better_description(image_bytes)
        new_descriptions.append(description)

    # Add the new descriptions to the DataFrame
    df["better_description"] = new_descriptions

    # Save the updated DataFrame
    df.to_csv(data_path, index=False)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Main function to add better descriptions to the train and test datasets.
    """
    add_better_description_to_dataset("data/train.csv")
    add_better_description_to_dataset("data/test.csv")


if __name__ == "__main__":
    main()
