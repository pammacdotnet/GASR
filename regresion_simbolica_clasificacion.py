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


def sigmoid(x):
    import math
    try:
        return 1 / (1 + math.exp(-x))
    except:
        return 1000

df = pd.read_csv('estrellas/star_classification.csv')

etiquetas = df['class'].to_numpy()

#1- Eliminamos las columnas que no aportan informacion util
df = df.drop(columns=['obj_ID','run_ID','rerun_ID','field_ID',
                     'spec_obj_ID','class','fiber_ID','MJD',
                     'cam_col','plate'])


list_terminals = df.columns

datos = df.to_numpy()

l = []
d = []

for v, e in zip(datos, etiquetas):
    if e == 'STAR' or e == 'QSO':
        d.append(v)
        if e == 'STAR':
            l.append(0)
        else:
            l.append(1)

datos = np.asarray(d)
etiquetas = np.asarray(l)

datos = datos[200:300]
etiquetas = etiquetas[200:300]

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
            f_uno, f_tres, f_dos, f_cuatro]

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
     y_pred.append(round(sigmoid(elem)))

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

