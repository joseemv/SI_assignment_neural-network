import numpy

# # Función asignada
# # Se deben cumplir estos criterios
# # T1(1, -1, 1, -1) == 1 else: 0
# T1 = a and -b and c and -d
# T2 = -a and -b and -c
# T3 = -(-a and b and d)
# T4 = (a and -b and -c and -d)
# fun = T1 or T2 or T3 or T4

def sigmoid(x):
    return 1/(1 + numpy.exp(-x))

print(input, sigmoid(b + numpy.sum(x*w)))

def main():
    # Genera la secuencia en binario desde 0000 hasta 1111
    # Obtiene cada combinación posible de unos y ceros para signar a las variables
    for n in range(15):
        # Transforma el número en binario
        x = format(n, '04b')
        a = int(x[0])
        b = int(x[1])
        c = int(x[2])
        d = int(x[3])

        # Genera los arrays
        # x -> vector de entrada
        x = numpy.asarray((a, b, c, d))
        # w -> vector que ajusta la precisión de la neurona
        w = numpy.asarray((a, b, c, d))
        # b = Bias -> desplaza el resultado para corregirlo
        b = 5
        numpy.sum(x+w)

        # Cada Tn representa una neurona
        T1 = sigmoid(b1 + numpy.sum(x*w1))
        T2 = sigmoid(b2 + numpy.sum(x*w2))
        T3 = sigmoid(b3 + numpy.sum(x*w3))
        T4 = sigmoid(b4 + numpy.sum(x*w4))
        # f -> función que agrega las neuronas
        f = sigmoid(bf + numpy.sum(numpy.asarray(T1, T2, T3, T4)*Wf))


if (__name__ == "__main__"):
    main()