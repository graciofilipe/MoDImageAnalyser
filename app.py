import streamlit as st
import streamlit.components.v1 as components
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import io
import json
import random
import traceback

# Get project ID from environment variable
PROJECT_ID = os.environ.get("GCP_PROJECT")
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

def generate_response(image_bytes, question):
    """Generates a response to a question about an image."""
    try:
        model = GenerativeModel("gemini-2.5-pro")
        image = Part.from_data(image_bytes, mime_type="image/jpeg")
        response = model.generate_content([image, question])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def plot_bounding_boxes(img, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.
    """
    st.write("Original bounding boxes received:", bounding_boxes)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray', 'beige',
    'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal', 'olive', 'coral',
    'lavender', 'violet', 'gold', 'silver'
    ] + [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

    bounding_boxes = parse_json(bounding_boxes)
    st.write("Parsed bounding boxes:", bounding_boxes)
    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    parsed_bounding_boxes = json.loads(bounding_boxes)
    st.write("Loaded bounding boxes JSON:", parsed_bounding_boxes)

    for i, bounding_box in enumerate(parsed_bounding_boxes):
        st.write(f"Processing bounding box {i}:", bounding_box)
        color = colors[i % len(colors)]
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)

        st.write(f"Coordinates for box {i}: y1={abs_y1}, x1={abs_x1}, y2={abs_y2}, x2={abs_x2}")

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)
    return img

st.title("Image Q&A with Gemini")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    image_bytes = uploaded_file.read()

    # Get the column width
    components.html(
        """
        <script>
        const stColumn = window.parent.document.querySelector('.main .block-container');
        const stColumnWidth = stColumn.clientWidth;
        window.parent.st.setSessionState({ 'column_width': stColumnWidth });
        </script>
        """,
        height=0,
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about the image"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            response = generate_response(image_bytes, prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    st.header("Object Detection")
    detection_prompt = st.text_input("Enter a prompt for object detection:")
    if st.button("Detect Objects"):
        if detection_prompt:
            with st.spinner("Detecting objects..."):
                try:
                    im = Image.open(io.BytesIO(image_bytes))
                    original_im = im.copy()
                    st.write("Original image size:", im.size)
                    im.thumbnail([640,640], Image.Resampling.LANCZOS)
                    st.write("Resized image size:", im.size)

                    model_name = "gemini-2.5-pro"
                    bounding_box_system_instructions = """
                        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
                        If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
                          """

                    # Convert the PIL image back to bytes
                    buffer = io.BytesIO()
                    im.save(buffer, format="PNG")
                    image_bytes_for_detection = buffer.getvalue()

                    image_part = Part.from_data(image_bytes_for_detection, mime_type="image/png")

                    model = GenerativeModel(model_name)
                    response = model.generate_content(
                        [bounding_box_system_instructions, detection_prompt, image_part],
                        generation_config={
                            "temperature": 0.5,
                        },
                    )

                    # Get the column width
                    column_width = st.session_state.get('column_width', 640)

                    # Resize the image to the column width
                    im.thumbnail([column_width, column_width], Image.Resampling.LANCZOS)

                    st.image(plot_bounding_boxes(im, response.text), caption="Detected Objects")
                except Exception as e:
                    st.error(f"An error occurred during object detection: {e}")
                    st.error(f"Full error: {traceback.format_exc()}")
        else:
            st.warning("Please enter a prompt for object detection.")
