import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# --- 1. Load the Saved Model ---
try:
    model = tf.keras.models.load_model('digit_recognizer_model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- 2. Create the Prediction Function ---
def predict_digit(sketchpad_data):
    if model is None or sketchpad_data is None:
        return "Model not loaded or no drawing provided."

    # Use the 'composite' key to get the final image with the drawing
    composite_image = sketchpad_data['composite']
    
    if composite_image is None:
        return "Please draw a digit."

    # --- ADVANCED PREPROCESSING ---
    
    # 1. Convert the NumPy array to a Pillow Image object, letting Pillow detect the mode
    pil_image = Image.fromarray(composite_image.astype('uint8')).convert('L')
    
    # 2. Invert colors (so digit is white, background is black)
    inverted_image = ImageOps.invert(pil_image)
    
    # 3. Find the bounding box of the digit
    bbox = inverted_image.getbbox()
    if bbox is None:
        return "Please draw a digit." # Handle blank image
        
    # 4. Crop the image to the bounding box
    cropped_image = inverted_image.crop(bbox)
    
    # 5. Resize the digit to fit within a 20x20 box to add padding, preserving aspect ratio
    cropped_image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    
    # 6. Create a new 28x28 black canvas and paste the digit in the center
    new_canvas = Image.new('L', (28, 28), 0) # Black background
    paste_x = (28 - cropped_image.width) // 2
    paste_y = (28 - cropped_image.height) // 2
    new_canvas.paste(cropped_image, (paste_x, paste_y))
    
    # 7. Convert the final canvas to a NumPy array for the model
    image = np.array(new_canvas)

    # 8. Reshape for the model and normalize.
    image = image.reshape(1, 28, 28).astype('float32') / 255.0

    # --- End of Preprocessing ---

    # Make a prediction
    prediction = model.predict(image)

    # Format the prediction into a dictionary of confidences
    confidences = {str(i): float(prediction[0][i]) for i in range(10)}
    
    return confidences

# --- 3. Create and Launch the Gradio Interface ---
inputs = gr.Sketchpad(image_mode='L')
outputs = gr.Label(num_top_classes=1, label="Prediction")

iface = gr.Interface(
    fn=predict_digit,
    inputs=inputs,
    outputs=outputs,
    title="Handwritten Digit Recognizer ✍️",
    description="Draw a digit from 0 to 9 on the canvas, and the model will predict what it is. For best results, draw a single, clear digit.",
    live=True
)

# Launch the app!
print("Launching the Gradio app...")
iface.launch()