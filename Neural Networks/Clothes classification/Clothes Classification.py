import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0           # instead of working with 0-255, better work with 0-1
test_images = test_images/255.0             # instead of working with 0-255, better work with 0-1

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),      #input layer; flatten takes the list and split it to make each pixel value a "neuron"
    keras.layers.Dense(128, activation="relu"),      #hidden layer; dense ==> fully connected layer; activation function take a linear model (1 degree) and make it polynomial (more than 1 degree)
    keras.layers.Dense(10, activation="softmax")     #output layer
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])   #loss functions minimize the functions in order to not make it too difficult to work with

model.fit(train_images, train_labels, epochs=5)        #here we train the model

prediction = model.predict(test_images)
for i in range(45,50):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual:"+ class_names[test_labels[i]])
    plt.title("Prediction:"+ class_names[np.argmax(prediction[i])])  #show the neuron which the highest value in terms of being certain that is true
    plt.show()
