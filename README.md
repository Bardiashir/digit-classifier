# digit-classifier
MNIST Digit Classifier (Keras)
This project is a simple neural-network classifier trained on the MNIST handwritten digits dataset using TensorFlow/Keras.
It demonstrates how to load image data, preprocess it, build a neural-network model, train it, and visualize predictions.

🚀 Project Overview

The goal of this project is to classify grayscale 28×28 handwritten digit images (0–9) using a fully connected neural network.

The model is trained using:

Flatten layer to convert images into 1D vectors

Dense hidden layers with ReLU activation

Softmax output layer for multi-class classification

Sparse categorical cross-entropy loss

Stochastic Gradient Descent (SGD) optimizer

🧠 Model Architecture
- Flatten (28×28 → 784)
- Dense layer: 300 neurons, ReLU
- Dense layer: 100 neurons, ReLU
- Dense output layer: 10 neurons, Softmax


The softmax layer converts outputs into a probability distribution across 10 digit classes.

📊 Dataset

The project uses the built-in Keras MNIST dataset, containing:

60,000 training images

10,000 test images

Pixel values normalized from 0–255 → 0–1

The README also visualizes the first 10 samples during execution.

🏋️ Training

The model is trained for:

epochs = 20
optimizer = SGD
loss = sparse_categorical_crossentropy

🔍 Prediction

After training, the script:

Predicts labels for the first 10 test images

Displays each image with its predicted digit

📦 Requirements

Install the required libraries:
```
tensorflow
matplotlib
numpy
```

(or use a venv and pip install -r requirements.txt)

▶️ How to Run
```
python main.py
```

(Make sure TensorFlow is installed.)
