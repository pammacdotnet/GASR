import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import Function
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

#Web ==> https://as595.github.io/classification/
#Dataset ==> conjunto de datos HTRU2
#Paper ==> https://figshare.com/articles/dataset/HTRU2/3080389/1

df = pd.read_csv('H:/Induccion fisica/articulo/Revista española de fisica/Datos pulsares/HTRU_2.csv') #pd.read_csv('./estrellas/pulsar.csv')


print(df.columns)
print('')
print ('Dataset has %d rows and %d columns including features and labels'%(df.shape[0],df.shape[1]))

etiquetas = df['class']

features = df.drop('class', axis=1)

list_terminals = features.columns


#Immplementamos la regresión simbolica

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
f_pi = Function.function('pi', 0, kernel.pi)
f_e = Function.function('e', 1, kernel.e)
f_pow = Function.function('^', 2, kernel.pow)
f_log = Function.function('L', 1, kernel.log)

functions = [f_div, f_sqrt, f_log, f_sub,
            f_uno, f_tres, f_dos, f_cuatro]

array_acc = []
array_kappa = []
cm_p = np.zeros((2,2))

array_arboles = []

k = 7
iterador = 0

for iterador in range(0, k):
    if iterador == 0:
        X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.75, random_state=66)
    elif iterador == 1:
        X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.85, random_state=66)
    elif  iterador == 2:
        X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.95, random_state=66)
    elif iterador == 3:
        X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.96, random_state=66)
    elif iterador == 4:
        X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.97, random_state=66)
    elif iterador == 5:
        X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.98, random_state=66)
    elif iterador == 6:
        X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.99, random_state=66)
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.99, random_state=66)
    
    X_smote = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    y_smote = y_train.to_numpy()

    ga1 = ga.GeneticAlgorithm(X_smote, y_smote, functions, list_terminals, 2, 
                            True, [], False, 500, 0.05, 25, False, True, function_activation, 1, 100)
    find = ga1.start()
    best_tree = ga1.getBestTree()
    array_arboles.append(best_tree)

    contador = 0

    r = []
    for i in range(0, len(y_test)):
            d = X_test[i]            
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
            if function_activation == 'sigmoid':
                y_pred.append(round(kernel.sigmoid(elem)))
            elif function_activation == 'tanh':
                y_pred.append(round(kernel.tanh(elem)))
            elif function_activation == 'leak_relu':
                y_pred.append(round(kernel.leaky_relu(elem)))
                
                
        else:
            y_pred.append(random.randint(0, 2))

    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_test, y_pred)
    k = cohen_kappa_score(y_pred, y_test)
    cm = confusion_matrix(y_test, y_pred)

    array_acc.append(acc)
    array_kappa.append(k)
    cm_p += cm

    contador += 1

    print('')
    print('    Iteracion ' + str(iterador))
    print('Accuracy : ' + str(acc*100) + '  -  Kappa : ' + str(k))
    print('')

print('')
print('Resultados promediados ')
print('')
print('Acc : ' + str(np.mean(array_acc)) + ' +/- ' + str(np.std(array_acc)) + '   - Kappa : ' + str(np.mean(array_kappa)))
print('')
print(cm_p/contador)

plt.plot(array_acc)
plt.show()
plt.plot(array_kappa)
plt.show()

input('Mostrar arboles obtenidos >>')

for elem in array_arboles:
    elem.displayTree()
    print('')




