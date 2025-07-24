import streamlit as st
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os

# Get project ID from environment variable
PROJECT_ID = os.environ.get("GCP_PROJECT")
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

def generate_description(image_bytes):
    """Generates a description for an image."""
    try:
        model = GenerativeModel("gemini-1.0-pro-vision")
        image = Part.from_data(image_bytes, mime_type="image/jpeg")
        response = model.generate_content([image, "Describe this image"])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

st.title("Image Describer")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    image_bytes = uploaded_file.read()
    with st.spinner("Generating description..."):
        description = generate_description(image_bytes)
        st.write(description)
