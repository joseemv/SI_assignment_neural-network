import numpy as np
from tensorflow import keras
from keras import layers
import time

#           TABLA DE LA VERDAD
#              --------------------------
#              |        Output          |
# ---------------------------------------  
# |  n | Input | T1 | T2 | T3 | T4 |  F |
# --------------------------------------
# |  0 | 0000  |  0 |  1 |  1 |  0 |  1 |
# |  1 | 0001  |  0 |  1 |  1 |  0 |  1 |
# |  2 | 0010  |  0 |  0 |  1 |  0 |  1 |
# |  3 | 0011  |  0 |  0 |  1 |  0 |  1 |
# |  4 | 0100  |  0 |  0 |  1 |  0 |  1 |
# |  5 | 0101  |  0 |  0 |  0 |  0 |  0 |
# |  6 | 0110  |  0 |  0 |  1 |  0 |  1 |
# |  7 | 0111  |  0 |  0 |  0 |  0 |  0 |
# |  8 | 1000  |  0 |  0 |  1 |  1 |  1 |
# |  9 | 1001  |  0 |  0 |  1 |  0 |  1 |
# | 10 | 1010  |  1 |  0 |  1 |  0 |  1 |
# | 11 | 1011  |  0 |  0 |  1 |  0 |  1 |
# | 12 | 1100  |  0 |  0 |  1 |  0 |  1 |
# | 13 | 1101  |  0 |  0 |  1 |  0 |  1 |
# | 14 | 1110  |  0 |  0 |  1 |  0 |  1 |
# | 15 | 1111  |  0 |  0 |  1 |  0 |  1 |
# ---------------------------------------

# Valores esperados por cada neurona
VALID_T1 = np.asarray(10)
VALID_T2 = np.asarray((0, 1))
VALID_T3 = np.asarray((0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15))
VALID_T4 = np.asarray(8)
VALID_F = np.asarray((0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15))

# Genera el vector de entradas y el vector de salidas esperadas para cada neurona y la función
def generate_arrays(input_bits):
    # Número de entradas posibles en función de los bits de entrada
    input_size = np.power(2, input_bits)
    # Inicializa los vectores de entradas y salidas esperadas
    input_array = np.zeros(shape=(input_size, input_bits))
    t1_expected_array = np.zeros(shape=(input_size, 1))
    t2_expected_array = np.zeros(shape=(input_size, 1))
    t3_expected_array = np.zeros(shape=(input_size, 1))
    t4_expected_array = np.zeros(shape=(input_size, 1))
    tf_expected_array = np.zeros(shape=(input_size, 1))
    
    # Genera la secuencia en binario desde 0000 hasta 1111
    for n in range(input_size):
        # Transforma el número en binario
        x = format(n, '04b')
        a = int(x[0])
        b = int(x[1])
        c = int(x[2])
        d = int(x[3])

        # Genera los datos de entrada
        input = np.asarray((a, b, c, d))
        # Actualiza el vector de datos de entrada
        input_array[n] = input
        
        # Actualiza los vectores de salidas esperadas
        t1_expected_array[n] = get_expected(n, 1)
        t2_expected_array[n] = get_expected(n, 2)
        t3_expected_array[n] = get_expected(n, 3)
        t4_expected_array[n] = get_expected(n, 4)
        tf_expected_array[n] = get_expected(n, "f")
        expected_arrays = (t1_expected_array, t2_expected_array, t3_expected_array, t4_expected_array, tf_expected_array)

    return input_array, expected_arrays

# Obtiene al valor de activación esperado para la entrada y neurona asignada
def get_expected(input, neuron):
    match neuron:
        case 1: valid_array = VALID_T1
        case 2: valid_array = VALID_T2
        case 3: valid_array = VALID_T3
        case 4: valid_array = VALID_T4
        case "f": valid_array = VALID_F

    expected = 0
    if (input in valid_array):
        expected = 1

    return expected

# Devuelve el resultado de la función para la tupla introducida
def forward(tuple):
    # Pesos
    t1_weight = np.asarray((90, -1000, 90, -1000))
    t2_weight = np.asarray((-1000, -1000, -1000, 0))
    t3_weight = np.asarray((1000, -14, -2, -14))
    t4_weight = np.asarray((1000, -1000, -1000, -1000))
    f_weight = np.asarray((1000, 1000, 1000, 1000))
    # Bias
    t1_bias = -100
    t2_bias = 20
    t3_bias = 22
    t4_bias = -100
    f_bias = -100
    
    t1_output = predict(tuple, t1_weight, t1_bias)
    t2_output = predict(tuple, t2_weight, t2_bias)
    t3_output = predict(tuple, t3_weight, t3_bias)
    t4_output = predict(tuple, t4_weight, t4_bias)
    neurons_output = np.asarray((t1_output, t2_output, t3_output, t4_output))
    f_output = predict(neurons_output, f_weight, f_bias)

    return f_output

# Predice la salida
def predict(input_array, weights, bias):
    # Producto de dos arrays -> dot(x, y) = x[0] * y[0] + x[1] * y[1],...
    activation = np.dot(input_array, weights) + bias

    return sigmoid(activation)

# Función sigmoidea utilizada en la útlima capa de la red neuronal. Devuelve entre 0 y 1
def sigmoid(activation):
    return 1/(1 + np.exp(-activation))

# Función de coste. Evalúa error cuadrático medio (MSE) de clasificación de la red
def calculate_error(expected_array, output_array):
    return np.sum(np.square(expected_array - output_array)) / (2*len(output_array))

# Demuestra el comportamiento de las neuronas en función de las entradas y los pesos y bias asignados
def train_manually():
    input_bits = 4
    # Vectores de salida
    t1_output = np.zeros(shape=(16, 1))
    t2_output = np.zeros(shape=(16, 1))
    t3_output = np.zeros(shape=(16, 1))
    t4_output = np.zeros(shape=(16, 1))
    f_output = np.zeros(shape=(16, 1))
    # Pesos
    t1_weight = np.asarray((90, -1000, 90, -1000))
    t2_weight = np.asarray((-1000, -1000, -1000, 0))
    t3_weight = np.asarray((1000, -14, -2, -14))
    t4_weight = np.asarray((1000, -1000, -1000, -1000))
    f_weight = np.asarray((1000, 1000, 1000, 1000))
    # Bias
    t1_bias = -100
    t2_bias = 20
    t3_bias = 22
    t4_bias = -100
    f_bias = -100
    
    # Genera las entradas y salidas esperadas
    input_array, expected_arrays = generate_arrays(input_bits)
    for n, input in enumerate(input_array):
        t1_output[n] = predict(input, t1_weight, t1_bias)
        t2_output[n] = predict(input, t2_weight, t2_bias)
        t3_output[n] = predict(input, t3_weight, t3_bias)
        t4_output[n] = predict(input, t4_weight, t4_bias)
        neurons_output = np.asarray((t1_output[n], t2_output[n], t3_output[n], t4_output[n])).T
        f_output[n] = predict(neurons_output, f_weight, f_bias)

    # Imprime los resultados
    with np.printoptions(suppress=True, precision=8):
        print("Salida T1")
        for input, expected, output in zip(input_array, expected_arrays[0], t1_output):
            print("Input:", input, "Expected:", expected, "Output:", output)

        print("Salida T2")
        for input, expected, output in zip(input_array, expected_arrays[1], t2_output):
            print("Input:", input, "Expected:", expected, "Output:", output)

        print("Salida T3")
        for input, expected, output in zip(input_array, expected_arrays[2], t3_output):
            print("Input:", input, "Expected:", expected, "Output:", output)
            
        print("Salida T4")
        for input, expected, output in zip(input_array, expected_arrays[3], t4_output):
            print("Input:", input, "Expected:", expected, "Output:", output)

        print("Salida función")
        for input, expected, output in zip(input_array, expected_arrays[4], f_output):
            print("Input:", input, "Expected:", expected, "Output:", output)

# Entrena las neuronas mediante Keras e imprime los resultados
def train_keras():
    input_bits = 4
    neurons_hidden_layers = 4
    neurons_output_layer = 1
    batch_size = 16
    epochs = 10000

    # Genera las entradas y salidas esperadas
    input_array, expected_arrays = generate_arrays(input_bits)
    # Selecciona la salida esperada de la función
    f_expected_array = expected_arrays[4]

    # Obtiene tiempo de inicio
    start_time = time.time()
    # Prepara el modelo de entrenamiento
    model = keras.Sequential(
        [
            keras.Input(shape=(input_bits)),
            layers.Dense(neurons_hidden_layers, activation="sigmoid"),
            layers.Dense(neurons_output_layer, activation="sigmoid")
        ]
    )
    model.compile(loss="mean_squared_error", optimizer="adam")
    # Entrena la red
    model.fit(input_array, f_expected_array, batch_size=batch_size, epochs=epochs)
    # Obtiene pesos y bias calculados
    weights_hidden_layers = model.layers[0].get_weights()[0]
    bias_hidden_layers = model.layers[0].get_weights()[1]
    weight_output_layer = model.layers[1].get_weights()[0]
    bias_output_layer = model.layers[1].get_weights()[1]
    # Predice la salida
    output_array = model.predict(input_array)
    # Calcula tiempo necesario para predecir la salida
    finish_time = time.time() - start_time

    # Imprime resultados
    print("\n")
    for input, expected, output in zip(input_array, f_expected_array, output_array):
        print("Input:", input, "Expected:", expected, "Output:", output)
    print("")
    print("Weights hidden layers:\n", weights_hidden_layers)
    print("Bias hidden layers:", bias_hidden_layers)
    print("")
    print("Weight output layer:\n", weight_output_layer)
    print("Bias output layer:", bias_output_layer)
    print("")
    print("Time:", finish_time)

def main():
    train_manually()
    train_keras()

if (__name__ == "__main__"):
    main()
