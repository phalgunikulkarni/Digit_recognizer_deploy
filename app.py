import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load the ONNX model
MODEL_PATH = "models/digit_model.onnx"
sess = ort.InferenceSession(MODEL_PATH)

# Get model input & output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
expected_shape = sess.get_inputs()[0].shape
st.write(f"Model expects input shape: {expected_shape}")

st.title("🖊️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below and the model will predict it.")

# Draw input
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data).astype(np.uint8))

    # Convert RGBA → Grayscale
    img = img.convert("L")
    img = ImageOps.invert(img)   # MNIST digits are white on black
    img = img.resize((28, 28))

    # Normalize & reshape
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.reshape(1, 28, 28)   # matches your training input

    # Run inference
    outputs = sess.run([output_name], {input_name: arr})[0]
    pred = np.argmax(outputs)

    st.image(img.resize((140, 140)), caption=f"Your drawn digit (processed)", width=140)
    st.subheader(f"Prediction: {pred}")
    st.bar_chart(outputs[0])
