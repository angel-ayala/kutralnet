import numpy as np

def normalize(data, interval=(0, 1)):
    a = interval[0]
    b = interval[1]
    minimo = np.min(data)
    maximo = np.max(data)
    return (b - a) * ((data - minimo) / (maximo - minimo)) + a
