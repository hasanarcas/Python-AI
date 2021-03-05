import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

word_index = data.get_word_index()                      #Word Mapping
word_index = {k:(v+3) for k,v in word_index.items()}    #Word Mapping
word_index["<PAD>"] = 0                                 #Word Mapping
word_index["<START>"] = 1                               #Word Mapping
word_index["<UNK>"] = 2                                 #Word Mapping
word_index["<UNUSED>"] = 3                              #Word Mapping
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])   #swap the keys and values

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)  #We want all the data to have the same length in order...
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)    #...to create an equal number of neurons for each dataset


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])   #if it doesn't find the value i it will write "?"

model = keras.Sequential()      # this is the same thing as writing keras.Sequential([]) and add all the layers at the beginning
model.add(keras.layers.Embedding(10000, 16))                #embedding layer is used for put in the same category the words with a near mean; here we create 10000 word vectors where each vector is composed of 16 dimensions(16 coeeficients)
model.add(keras.layers.GlobalAvgPool1D())                   #GlobalAvgPool1D layer is used to shrink the 16 dimensions to a lower dimension number
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))      #The final result will be either 0 or 1, either Bad or Good

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]        #validation data
x_train = train_data[10000:]
y_val = train_labels[:10000]      #validation data
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)         #batch_size = the number of review that the program will take each time; verbose is just used for the animation while loading the epochs

results = model.evaluate(test_data, test_labels)

test_review = test_data[2]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))