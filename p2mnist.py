from tensorflow import keras
from keras import layers
import random
import time

def main(training_sample=60000, validation_sample=10000):
    # Parámetros
    output_classes = 10
    batch_size = 128
    epochs = 15
    validation_split = 0.1

    # Obtiene conjunto de entrenamiento y de validación de la base de datos mnist
    (input_training_set, label_training_set), (input_validation_set, label_validation_set) = keras.datasets.mnist.load_data()

    # Convierte las entradas en un único vector de 784 elementos. (60000, 28*28)
    input_training_set = input_training_set.reshape(input_training_set.shape[0], input_training_set.shape[1]*input_training_set.shape[2])
    input_validation_set = input_validation_set.reshape(input_validation_set.shape[0], input_validation_set.shape[1]*input_validation_set.shape[2])

    # Normaliza las imágenes
    input_training_set = input_training_set/255
    input_validation_set = input_validation_set/255

    # Convierte en un vector de dimensión 10
    label_training_set = keras.utils.to_categorical(label_training_set, output_classes)
    label_validation_set = keras.utils.to_categorical(label_validation_set, output_classes)

    # Selecciona una muestra aleatoria
    random_indices_training = random.sample(range(input_training_set.shape[0]), training_sample)
    input_training_set_sample = input_training_set[random_indices_training]
    label_training_set_sample = label_training_set[random_indices_training]
    random_indices_validation = random.sample(range(input_validation_set.shape[0]), validation_sample)
    input_validation_set_sample = input_validation_set[random_indices_validation]
    label_validation_set_sample = label_validation_set[random_indices_validation]

    start_time = time.time()
    model = keras.Sequential(
        [
            keras.Input(shape=784),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax")
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(input_training_set_sample, label_training_set_sample, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    # Evalúa la salida
    output_training = model.evaluate(input_training_set_sample, label_training_set_sample)
    output_validation = model.evaluate(input_validation_set_sample, label_validation_set_sample)
    finish_time = time.time() - start_time

    print("Resultados entrenamiento")
    print("Loss:", output_training[0])
    print("Precisión:", output_training[1])
    print("\nResultados validación")
    print("Loss:", output_validation[0])
    print("Precisión:", output_validation[1])
    print("Time:", finish_time, "\n")

if (__name__ == "__main__"):
    main()
