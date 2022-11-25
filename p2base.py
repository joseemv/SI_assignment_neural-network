import numpy as np

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

# Genera los arrays de entradas y salidas esperadas
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

# Predice la salida
def predict(input_array, weigths, bias):
    # Producto de dos arrays -> dot(x, y) = x[0] * y[0] + x[1] * y[1],...
    activation = np.dot(input_array, weigths) + bias

    return sigmoid(activation)

# Función sigmoidea utilizada en la útlima capa de la red neuronal. Devuelve entre 0 y 1
def sigmoid(activation):
    return 1/(1 + np.exp(-activation))

# Función de coste. Evalúa error cuadrático medio (MSE) de clasificación de la red
def calculate_error(expected_array, output_array):
    return np.sum(np.square(expected_array - output_array)) / (2*len(output_array))

# Demuestra el comportamiento de las neuronas en función de las entradas y los pesos y bias asignados
def test_neuronas():
    # Vectores de salida
    t1_output = np.zeros(shape=(16, 1))
    t2_output = np.zeros(shape=(16, 1))
    t3_output = np.zeros(shape=(16, 1))
    t4_output = np.zeros(shape=(16, 1))
    tf_output = np.zeros(shape=(16, 1))
    # Pesos
    t1_weigth = np.asarray((90, -1000, 90, -1000))
    t2_weigth = np.asarray((-1000, -1000, -1000, 0))
    t3_weigth = np.asarray((1000, -14, -2, -14))
    t4_weigth = np.asarray((1000, -1000, -1000, -1000))
    tf_weigth = np.asarray((1000, -20, -2, -20))
    # Bias
    t1_bias = -100
    t2_bias = 20
    t3_bias = 22
    t4_bias = -100
    tf_bias = 30

    # Genera la secuencia en binario desde 0000 hasta 1111
    for n in range(16):
        # Transforma el número en binario
        x = format(n, '04b')
        a = int(x[0])
        b = int(x[1])
        c = int(x[2])
        d = int(x[3])
        # Genera los datos de entrada
        input = np.asarray((a, b, c, d))

        # Predice el resultado
        t1_output[n] = predict(input, t1_weigth, t1_bias)
        t2_output[n] = predict(input, t2_weigth, t2_bias)
        t3_output[n] = predict(input, t3_weigth, t3_bias)
        t3_output[n] = predict(input, t4_weigth, t4_bias)
        tf_output[n] = predict(input, tf_weigth, tf_bias)

    # f = sigmoidea(bias_f + np.sum(np.asarray((T1,T2,T3,T4)*peso_f)))

    # Imprime los resultados
    with np.printoptions(suppress=True, precision=8):
        print("Salida T1")
        for i in range(16):
            print("Input:", format(i, '04b'), "| Output:", t1_output[i])

        print("Salida T2")
        for i in range(16):
            print("Input:", format(i, '04b'), "| Output:", t2_output[i])

        print("Salida T3")
        for i in range(16):
            print("Input:", format(i, '04b'), "| Output:", t3_output[i])

        print("Salida T4")
        for i in range(16):
            print("Input:", format(i, '04b'), "| Output:", t4_output[i])

        print("Salida función")
        for i in range(16):
            print("Input:", format(i, '04b'), "| Output:", tf_output[i])

def main():
    test_neuronas()

if (__name__ == "__main__"):
    main()
