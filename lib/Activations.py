import numpy as np
import math


# Activation funcs hyperbolic tangent and its derivative
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1-np.tanh(x)**2


# Activation funcs sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + math.e ** (-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Activation funcs softmax and its derivative
def softmax(x):
    # print(f'x = {x}')
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))
