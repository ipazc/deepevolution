import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from deepevolution import wrap_keras

# We enable the deep evolution API on keras models: fit_evolve()
wrap_keras()


# Build the keras model
inp = Input((28, 28))
layer = Flatten()(inp)
layer = Dense(30, activation="relu")(layer)
layer = Dense(10, activation="softmax")(layer)
model = Model(inp, layer)

model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Pick the training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

# Evolve the model. By default, it evolves the negative loss
history = model.fit_evolve(x_train, y_train, max_generations=10, population=10, top_k=3)

print(f"Model accuracy: {model.evaluate(x_train, y_train, batch_size=2048)[1]}")

pd.DataFrame(history).plot()
plt.show()
