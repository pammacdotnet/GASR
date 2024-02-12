import pandas as pd
import numpy as np

import unidades
import Tree
import kernel
import Function
import GeneticAlgorithm as ga
from sklearn.model_selection import train_test_split


'''
    Preparamos el algoritmo genetico
'''
#1- Definicion de las funciones a utilizar

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

functions = [f_sub, f_add, f_div, f_log, f_e]

'''
    Preparamos el dataset con los datos de diabetis
'''

df = pd.read_csv('./diabetis/diabetes.csv')

goal = df['DiabetesPedigreeFunction']

list_terminals = df.columns

columns_to_keep = df.columns.drop('DiabetesPedigreeFunction')
columns_to_keep = columns_to_keep.drop('Outcome')
df_filtered = df[columns_to_keep]

list_terminals = columns_to_keep.to_numpy()

datos = df_filtered.to_numpy()
goal = goal.to_numpy()



X_train, X_test, y_train, y_test = train_test_split(datos, goal, 
                                    test_size=0.33, random_state=42)

ga1 = ga.GeneticAlgorithm(X_train, y_train, functions, list_terminals, 2, True, [], False, 15, 0.15, 15000)
find = ga1.start()
best_tree = ga1.getBestTree()