# Handwritten Digit Recognizer ✍️

A classic machine learning application that recognizes handwritten digits from 0 to 9. This project uses a neural network trained on the MNIST dataset and provides a simple web interface where you can draw a digit and get a real-time prediction.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Piyushcode7804/Handwritten-digit-recognizer)




## Features

* **Real-time Prediction:** Draw a digit on the canvas and see the model's prediction instantly.
* **High Accuracy:** The model is trained using data augmentation for improved robustness and achieves high accuracy on the MNIST test set.
* **Advanced Preprocessing:** User drawings are automatically centered, cropped, and resized to match the format of the MNIST dataset, significantly improving real-world performance.
* **Interactive UI:** Built with Gradio for a simple and intuitive user experience.
* **Deployed and Sharable:** Easily deployed on Hugging Face Spaces.

***

## Technologies Used

* **Python**
* **TensorFlow / Keras:** For building and training the neural network.
* **Gradio:** For creating the interactive web UI.
* **Pillow (PIL):** For advanced image processing.
* **NumPy:** For numerical operations.
* **Matplotlib:** For visualizing the dataset and training history.

***

## Project Structure

```
├── digit_recognizer_model.keras            # The trained and saved model
├── handwritten-digit-recognizer.py         # Script to train the model
├── app.py                                  # The Gradio web application script
├── requirements.txt                        # Python dependencies for deployment
└── README.md                               # Project documentation
```
***

## Local Setup and Installation

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Piyushthourani/Handwritten-Digit-Recognition.git
    cd Handwritten-Digit-Recognition
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

***

## Usage

1.  **(Optional) Train the Model:**
    If you want to train the model from scratch, run the `handwritten-digit-recognizer.py` script. This will train a new model and save it as `digit_recognizer_model.keras`.
    

2.  **Launch the Web App:**
    To start the user interface, run the `app.py` script.
    ```bash
    python app.py
    ```
    Open the local URL printed in your terminal (e.g., `http://127.0.0.1:7860`) in your web browser to start drawing!

***

## Deployment

This application is designed for easy deployment on [Hugging Face Spaces](https://huggingface.co/spaces). Simply create a new Gradio Space and upload the following files:
* `app.py`
* `digit_recognizer_model.keras`
* `requirements.txt`

Hugging Face will automatically handle the setup and deployment.
