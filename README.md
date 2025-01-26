# Pokemon Project

This project focuses on generating detailed and engaging descriptions for Pokémon images. The primary objective is to entertain users with vivid and creative narratives while simultaneously exploring the potential of modern neural network models in image understanding and caption generation.

By leveraging cutting-edge AI technology, this project aims to:

* Demonstrate the capabilities of neural networks in visual recognition and language generation.

* Delight Pokémon enthusiasts with imaginative and descriptive interpretations of their favorite characters.

* Push the boundaries of AI creativity by crafting descriptions that resonate with human audiences.

Ultimately, this project bridges the gap between entertainment and AI innovation, providing a fun and insightful experience for all participants.

# Dataset

We utilized the following dataset:https://huggingface.co/datasets/svjack/pokemon-blip-captions-en-zh

The original dataset size was reduced from 833 entries to 200 for this project.

Image descriptions were rephrased using LangChain to enhance quality and ensure consistency.

## Preprocessing

To improve training quality, we used the following prompt to create enriched descriptions for our dataset:

```
You will be provided with a Pokémon image.
Your task is to create a short description of the Pokémon image.

Example of your output:
This character resembles a fantasy creature that looks like a combination of a mineral
and a living being. He has blue eyes, a large white “mustache,” a pointed crystal on his head,
and a stone body with flecks of ice crystals.
```

**Example Result**:

Original Description:a blue and white dragon with its mouth open

Rephrased Description:This character is a serpent-like dragon Pokémon, featuring a long, elongated body predominantly colored blue and cream. It has fierce red eyes, a wide-open mouth revealing sharp teeth, and prominent spiky fins along its head and sides. Its body is decorated with a pattern of blue and cream stripes, giving it a powerful and intimidating appearance.

# Model

**Base Model:**

https://huggingface.co/Salesforce/blip-image-captioning-large

**Fine-tuned Model:**

https://huggingface.co/RidzIn/Pokemon-Describer

For our project, we selected a compact BLIP model with 470M parameters due to its efficiency and compatibility with limited computational resources. Despite its advantages, the model has a restricted context window, which influenced the final results.

## Training Process

We employed HuggingFace's Trainer class for fine-tuning the model. Below is an example of the training setup:
```
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```
The training process involved minimizing the gap between generated and ground-truth captions by optimizing the model on our curated dataset.

# Metrics

To evaluate the model's performance, we utilized the following metric:https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

This sentence-transformers model maps sentences and paragraphs to a 384-dimensional dense vector space. It was used in this project for tasks like clustering and semantic search, specifically to measure similarity between ground-truth captions and model-generated descriptions.

**Results:**

Achieved 68% similarity score.

Considering the context window limitations and the fact that training samples often exceeded the model's capacity, this score is a strong indicator of the model's ability to generate meaningful descriptions.

# Desktop Aplication description
This project is a desktop application built with Python that leverages machine learning to generate textual descriptions of Pokemon from uploaded images. The application includes a graphical user interface (GUI) for user interaction, allowing users to upload RGB images of Pokemon in JPG or PNG formats and generate descriptions using a pre-trained machine learning model. The application is packaged as a Docker container to ensure portability and ease of deployment.

## Features
1. Upload Images: Users can upload RGB images of Pokemon in JPG or PNG formats.

2. Generate Descriptions: A pre-trained machine learning model generates textual descriptions of the Pokemon in the uploaded image.

3. User-Friendly Interface: The application uses a GUI for user interaction.

4. Dockerized Environment: The application is containerized with Docker to ensure full portability and platform independence.

## System Requirements

* Supported Operating Systems: Windows, macOS, Linux

* Docker

## Libraries

The following Python libraries are meant to be used in this project:

Core libraries

1. Tkinter: creating the graphical user interface.

    o Provides an easy-to-use framework for building desktop applications.

2. Pillow: image loading and processing.

    o Allows reading and processing JPG and PNG images.

3. (Tensorflow?, PyTorch?): running the pre-trained machine learning model

4. Numpy: potential image data storing and processing

## Application Usage

GUI Components

* Upload Button: Opens a file dialog to select an image of a Pokemon (JPG or PNG).

* Generate Description Button: Initiates the process of generating the Pokemon description.

* Output Box: Displays the generated description.

Workflow

1. Launch the application.

2. Click the Upload button and select an image of a Pokemon.

3. Click the Generate Description button.

4. View the generated description in the output box.


# Streamlit UI

![image](https://github.com/user-attachments/assets/1dd7c665-dad4-4d70-837c-193073927b80)
![image](https://github.com/user-attachments/assets/08985cd7-9b31-4793-9cd2-b55a309b32a8)
