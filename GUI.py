import streamlit as st
from PIL import Image
from model_api import invoke_model

# -----------------------------------------------------------------------------
# Description Generation
# -----------------------------------------------------------------------------
def generate_description(image_path):
    """
    Generates a textual description for a given Pokémon image using the
    'RidzIn/Pokemon-Describer' model.

    Args:
        image_path (str): The path to the image file on the local system.

    Returns:
        str:
            The generated description if successful, or an error message if any
            exception is encountered.
    """
    try:
        result = invoke_model(model_name="RidzIn/Pokemon-Describer", image_path=image_path)
        return result["generated_description"]
    except Exception as e:
        return f"Error: {str(e)}"


# -----------------------------------------------------------------------------
# Streamlit Application
# -----------------------------------------------------------------------------
st.title("Pokemon Image Describer")

# File uploader for PNG/JPG images
uploaded_file = st.file_uploader("Upload a Pokemon image", type=["png", "jpg"])

# Once a file is uploaded, display the image and provide a description generator
if uploaded_file:
    # Convert the uploaded image to RGB before displaying
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image locally to a temporary file
    image_path = "image.png"
    image.save(image_path)

    # Button triggers the model to generate a description
    if st.button("Generate Description"):
        description = generate_description(image_path)
        st.write("Description:")
        st.write(description)
