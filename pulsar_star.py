import math
import matplotlib.pyplot as plt
import pandas as pd
import Function
import Tree
import GeneticAlgorithm as ga
import numpy as np
import kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import random

def sigmoid(x):
    import math
    try:
        return 1 / (1 + math.exp(-x))
    except:
        return 1000
    
df = pd.read_csv('estrellas/pulsar_data_train.csv')

etiquetas = df['target_class'].to_numpy()

'''
    1. Mean of the integrated profile.
    2. Standard deviation of the integrated profile.
    3. Excess kurtosis of the integrated profile.
    4. Skewness of the integrated profile.
    5. Mean of the DM-SNR curve.
    6. Standard deviation of the DM-SNR curve.
    7. Excess kurtosis of the DM-SNR curve.
    8. Skewness of the DM-SNR curve.
    9. Class : 0 no es un pulsar, 1 si es un pulsar
'''

#1- Eliminamos las columnas que no aportan informacion util
df = df.drop(columns=['target_class'])

list_terminals = df.columns

# Aplicar normalización min-max a todas las columnas
for column in df.columns:
    min_value = df[column].min()
    max_value = df[column].max()
    df[column + '_Normalizada'] = (df[column] - min_value) / (max_value - min_value)


datos = df.to_numpy()

datos = datos[0:200]
etiquetas = etiquetas[0:200]

x_train, x_test, y_train, y_test = train_test_split(datos, etiquetas, test_size=0.25)


#Immplementamos la regresión simbolica

f_add = Function.function('+', 2, kernel.add)
f_sub = Function.function('-', 2, kernel.sub)
f_plus = Function.function('*', 2, kernel.plus)
f_div = Function.function('/', 2, kernel.div)
f_sqrt = Function.function('Q', 1, kernel.Q)
f_cos = Function.function('cos', 1, kernel.cos)
f_sen = Function.function('sen', 1, kernel.sin)
f_div2 = Function.function('/', 2, kernel.protected_div)
f_dos = Function.function('2', 0, kernel.two)
f_uno = Function.function('1', 0, kernel.one)
f_tres = Function.function('3', 0, kernel.three)
f_cuatro = Function.function('4', 0, kernel.four)
f_pi = Function.function('pi', 0, kernel.pi)
f_e = Function.function('e', 1, kernel.e)
f_pow = Function.function('^', 2, kernel.pow)
f_log = Function.function('L', 1, kernel.log)

functions = [f_add, f_div, f_sqrt, f_log, f_sub,
            f_uno, f_tres, f_dos, f_cuatro, f_cos, f_sen]

ga1 = ga.GeneticAlgorithm(x_train, y_train, functions, list_terminals, 2, 
                          True, [], False, 15, 0.05, 10000, True, True)
find = ga1.start()
best_tree = ga1.getBestTree()

#Debemos testear el sistema con los datos de test

r = []
for i in range(0, len(y_test)):
        d = x_test[i]            
        g = y_test[i]
        dict = {}
        for j in range(0, len(list_terminals)):
            name = list_terminals[j]
            value = d[j]
            dict.update({name:value})

        result = best_tree.evaluate(best_tree.getRoot(), dict)
        r.append(result)

r = np.asarray(r)
y_pred = []

for elem in r:
    if not elem is None:
         y_pred.append(round(sigmoid(elem)))
    else:
         y_pred.append(random.randint(0, 2))

y_pred = np.asarray(y_pred)
acc = accuracy_score(y_test, y_pred)
k = cohen_kappa_score(y_pred, y_test)
cm = confusion_matrix(y_test, y_pred)

print('')
print('    Resultados : ')
print('')
print('Accuracy : ', acc*100)
print('Kappa : ', k)
print('')
print(cm)

'''for elem, label in zip(r, y_test):
     print('Real : ', label)   
     print('Sigmoid : ', sigmoid(elem))
     print('Predicted : ', round(sigmoid(elem)))
     input('>>')'''


