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

def generate_response(image_bytes, question):
    """Generates a response to a question about an image."""
    try:
        model = GenerativeModel("gemini-2.5-flash")
        image = Part.from_data(image_bytes, mime_type="image/jpeg")
        response = model.generate_content([image, question])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

st.title("Image Q&A with Gemini")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    image_bytes = uploaded_file.read()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about the image"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            response = generate_response(image_bytes, prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
