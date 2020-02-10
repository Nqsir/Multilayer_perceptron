import numpy as np


# MSE loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


# Cross_entropy loss function and its derivative
def cross_entropy(y, s):
    """Return the cross-entropy of vectors y and s.
    :type y: ndarray
    :param y: one-hot vector encoding correct class
    :type s: ndarray
    :param s: softmax vector
    :returns: scalar cost
    """
    return -(np.log(s[np.where(y)]).mean())


def cross_entropy_prime_softmax(y, s):
    """Return the sensitivity of cross-entropy cost to input of softmax.
    :type y: ndarray
    :param y: one-hot vector encoding correct class
    :type s: ndarray
    :param s: softmax vector
    :returns: ndarray of size len(s)
    """
    return s - y
