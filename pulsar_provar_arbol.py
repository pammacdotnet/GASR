import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import Function
import Tree
import node
import GeneticAlgorithm as ga
import numpy as np
import kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import random
import kernel

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

function_activation = 'sigmoid'

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
f_ten = Function.function('10', 0, kernel.ten)
f_pi = Function.function('pi', 0, kernel.pi)
f_e = Function.function('e', 1, kernel.e)
f_pow = Function.function('^', 2, kernel.pow)
f_log = Function.function('L', 1, kernel.log)

functions = [f_div, f_sqrt, f_log, f_sub,
            f_uno, f_tres, f_dos, f_cuatro]

df = pd.read_csv('H:/Induccion fisica/articulo/Revista española de fisica/Datos pulsares/HTRU_2.csv') #pd.read_csv('./estrellas/pulsar.csv')

print(df.columns)
print('')
print ('Dataset has %d rows and %d columns including features and labels'%(df.shape[0],df.shape[1]))

etiquetas = df['class']

features = df.drop('class', axis=1)
list_terminals = features.columns

features = features.to_numpy()
etiquetas = etiquetas.to_numpy()

print(features.shape)

print(df.columns)
print('')
print ('Dataset has %d rows and %d columns including features and labels'%(df.shape[0],df.shape[1]))


t1 = Tree.Tree(functions, list_terminals)
n1 = node.node(f_sub, 'function')
n2 = node.node('ex_kurt_pf','var')
n3 = node.node(f_uno,'function')
n4 = node.node(f_sqrt, 'function')
n5 = node.node(f_log, 'function')
n6 = node.node(f_cuatro, 'function')
n1.setChild(n2)
n5.setChild(n6)
n4.setChild(n5)
n1.setChild(n4)
t1.root = n1
t1.displayTree()

input('>>')

r = []
for i in range(0, len(etiquetas)):
        d = features[i]            
        g = etiquetas[i]
        dict = {}
        for j in range(0, len(list_terminals)):
            name = list_terminals[j]
            value = d[j]
            dict.update({name:value})

        result = t1.evaluate(t1.getRoot(), dict)
        r.append(result)

r = np.asarray(r)
y_pred = []

for elem in r:
    if not elem is None:
        if function_activation == 'sigmoid':
            y_pred.append(round(kernel.sigmoid(elem)))
        elif function_activation == 'tanh':
            y_pred.append(round(kernel.tanh(elem)))
        elif function_activation == 'leak_relu':
            y_pred.append(round(kernel.leaky_relu(elem)))
             
             
    else:
         y_pred.append(random.randint(0, 1))

y_pred = np.asarray(y_pred)
acc = accuracy_score(etiquetas, y_pred)
k = cohen_kappa_score(y_pred, etiquetas)
cm = confusion_matrix(etiquetas, y_pred)

print('')
print('    Resultados : ')
print('')
print('Accuracy : ', acc*100)
print('Kappa : ', k)
print('')
print(cm)


