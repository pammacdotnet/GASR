import math
import numpy as np

def add(a, b):
    return a+b

def sub(a, b):
    return a-b

def abs(a):
    if a > 0:
        return a
    else:
        return -a

def plus(a, b):
    return a*b

def div(a, b):
    if b == 0:
        return 1e-6
    else:
        return a/b

def protected_div(a, b):
    if abs(b) < 1e-6:
        return 1
    return a / b    

def sin(a):
    return math.sin(a)

def cos(a):
    return math.cos(a)

def Q(a):
    return math.sqrt(a)

def one():
    return 1

def two():
    return 2

def three():
    return 3

def four():
    return 4

def five():
    return 5

def ten():
    return 10

def pi():
    return math.pi

def e(a):
    return math.exp(a)

def pow(x, y):
    return math.pow(x, y)

def log(a):
    return math.log(a)


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        return 1000


def tanh(z):
    ''' Activacion tangente hiperbolica '''
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
