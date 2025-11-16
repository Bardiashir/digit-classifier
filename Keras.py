
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np


# From Keras, we load the MNIST digits classification dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

plt.figure(figsize=(5, 2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
plt.show()

print("In the training set, there are", x_train.shape[0], "instances (2D grayscale image data with 28×28 pixels. \
In turn, every image is represented as a 28×28 array rather than a 1D array of size 784. \
Pixel values range from 0 (white) to 255 (black).) \
The associated labels are digits ranging from 0 to 9.")

# We scale the input feature down to 0-1 values, by dividing them by 255.0
x_train = x_train / 255.0
x_test = x_test / 255.0

# Then we create a Sequential model. A sequential model is a stack of layers connected sequentially.
# This is the simplest kind of model for neural networks in Keras.
model = keras.Sequential()

# we build a first layer to the model, that will convert each 2D image into a 1D array.
# For this, add a 'Flatten layer', and specify the shape (input_shape) of the instances [28,28].
model.add(keras.layers.Flatten(input_shape=(28, 28)))

# Build the first hidden layer to the model.
# For this, use a 'Dense layer' with 300 neurons, and use the ReLU activation function.
# A dense Layer is simple layer of neurons in which each neuron receives input from all the neurons of previous layer.
model.add(keras.layers.Dense(300, activation="relu"))

# Build a second hidden layer to the model.
# For this,we use a 'Dense layer' with 100 neurons, also using the ReLU activation function.
model.add(keras.layers.Dense(100, activation="relu"))

# Build an output layer to the model.
# For this,we use a 'Dense layer' with 10 neurons (one per class), using the softmax activation function.
model.add(keras.layers.Dense(10, activation="softmax"))

print("The softmax action function was use for the output layer because it converts the raw outputs into a probability \
distribution over the 10 classes, with all probabilities non-negative and summing to 1, which is ideal for multi-class classification.")


print("The size of the first hidden layer is(None, 300). None means the the batch size dimension can be any value.\
The total number of parameters of the first hidden layer is 235500, which refers to the number of trainable weights and biases in that layer")
model.summary()

# Then we call the method compile() on our model to specify the loss function and the optimizer to use.
# Set the loss function to be "sparse_categorical_crossentropy" and use the stochastic gradient descent optimizer.
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd")

print("An epoch is one complete pass of the learning algorithm through the entire training dataset.")


model.fit(x_train, y_train,
          epochs=20)


# Then we test the model: use the method predict() to predict the labels of the first 10 instances of the test set
plt.close('all')
y_pred = model.predict(x_test[:10])
plt.figure(figsize=(5, 2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.title('Predicted label: ' +
              str(np.argmax(y_pred[i])))
    plt.imshow(x_test[i], cmap='gray')
    plt.axis('off')
plt.show()
