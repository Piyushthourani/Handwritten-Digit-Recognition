import gradio as gr
import tensorflow as tf
import numpy as np

# --- 1. Load the Saved Model ---
try:
    model = tf.keras.models.load_model('digit_recognizer_model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- 2. Create the Prediction Function ---
def predict_digit(image):
    if model is None:
        return "Model not loaded. Please check the console for errors."

    # Preprocess the image:
    # 1. Invert colors manually (black on white -> white on black)
    image = 255 - image
    # 2. Reshape for the model and normalize pixel values.
    image = image.reshape(1, 28, 28).astype('float32') / 255.0

    # Make a prediction
    prediction = model.predict(image)

    # Format the prediction into a dictionary of confidences
    confidences = {str(i): float(prediction[0][i]) for i in range(10)}
    
    return confidences

# --- 3. Create and Launch the Gradio Interface ---
# Define the input component (with all old arguments removed)
inputs = gr.Sketchpad(
    image_mode='L' # 'L' for grayscale
)

# Define the output component
outputs = gr.Label(num_top_classes=3, label="Predictions")

# Build the interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=inputs,
    outputs=outputs,
    title="Handwritten Digit Recognizer ✍️",
    description="Draw a digit from 0 to 9 on the canvas, and the model will predict what it is.",
    live=True
)

# Launch the app!
print("Launching the Gradio app...")
iface.launch()