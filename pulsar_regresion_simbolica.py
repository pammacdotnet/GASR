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


'''features = features.to_numpy()
etiquetas = etiquetas.to_numpy()'''



X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.99, random_state=45) #66


'''# rebalance using random upsampling
X_oversampled, y_oversampled = resample(
  X_train[y_train == 1],
  y_train[y_train == 1],
  replace = True,
  n_samples = len(y_train[y_train == 0]),
  random_state = 0
)

# put it back together
X_resample = pd.DataFrame(
  np.vstack((X_train[y_train == 0], X_oversampled)),
  columns=X_train.columns
)

y_resample = pd.Series(
  np.hstack((y_train[y_train == 0], y_oversampled))
).reset_index(drop=True)


# now for SMOTE
sm = SMOTE(random_state=0)
X_smote, y_smote = sm.fit_resample(X_train, y_train)

X_smote = pd.DataFrame(X_smote,columns=list_terminals)
y_smote = pd.Series(y_smote)

elementos, conteo = np.unique(y_smote, return_counts=True)
print('X : ', X_smote.shape)
for elemento, count in zip(elementos, conteo):
    print(f"Elemento: {elemento}, Instancias: {count}")'''

print('')
print('Train:')
elementos, conteo = np.unique(y_train, return_counts=True)
for elemento, count in zip(elementos, conteo):
     print(f"Elemento: {elemento}, Instancias: {count}")

print('')
print('Test:')
elementos, conteo = np.unique(y_test, return_counts=True)
for elemento, count in zip(elementos, conteo):
     print(f"Elemento: {elemento}, Instancias: {count}")

'''print('')
input('Continuar >>')'''

X_smote = X_train.to_numpy() #X_smote.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

y_smote = y_train.to_numpy()


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
f_cinco = Function.function('4', 0, kernel.five)
f_diez = Function.function('4', 0, kernel.ten)
f_pi = Function.function('pi', 0, kernel.pi)
f_e = Function.function('e', 1, kernel.e)
f_pow = Function.function('^', 2, kernel.pow)
f_log = Function.function('L', 1, kernel.log)

functions = [f_div, f_sqrt, f_log, f_sub,
            f_uno, f_tres, f_dos, f_cuatro, f_cinco, f_diez]

ga1 = ga.GeneticAlgorithm(X_smote, y_smote, functions, list_terminals, 2, 
                          True, [], False, 25, 0.05, 10000, True, True, function_activation, 0.85, 100000)
find = ga1.start()
best_tree = ga1.getBestTree()

#Debemos testear el sistema con los datos de test

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

print('')
print('    Resultados : ')
print('')
print('Accuracy : ', acc*100)
print('Kappa : ', k)
print('')
print(cm)

'''for elem, label in zip(r, y_test):
     print('Real : ', label)   
     print('Sigmoid : ', kernel.sigmoid(elem))
     print('Predicted : ', round(kernel.sigmoid(elem)))
     input('>>')'''


