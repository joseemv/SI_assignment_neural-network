import numpy as np
from tensorflow import keras
from keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.Sequential(
    [
        keras.Input(shape=784),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ]
)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)
# Obtiene pesos y bias calculados
weigth_layers = model.layers[0].get_weights()[0]
bias_layers = model.layers[0].get_weights()[1]
# Predice la salida
# output_array = model.predict(input_array)

# Imprime resultados
# for input, output in zip(input_array, output_array):
#     print("Input:", input, "Output:", output)
print("Weights:", weigth_layers)
print("Bias:", bias_layers)