import kernel
import Function
import Tree
import pickle as pkl
import GeneticAlgorithm as ga
import pandas as pd
import numpy as np
import math
from json_tricks import dumps, load, loads
import csv_experiment


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

functions = [f_plus, f_div2]

file_csv = './experimentos/resistencia_cable.csv'
index_goal = 0

datos = csv_experiment.get_data_experiment(file_csv)
atributos = csv_experiment.get_atributes(file_csv)

print(atributos)
at = atributos.copy()
print('Valor objetivo : ', atributos[index_goal])


atributos = np.delete(atributos, index_goal).to_numpy()
    
#Aplicamos un factor de escala

Y = datos[:,index_goal]

datos = np.delete(datos, (index_goal), axis=1)

try:
    fileObject2 = open('./memory.pkl', 'rb')
    memory = pkl.load(fileObject2)
    fileObject2.close()
    use_memory = True
except:
    memory = []
    use_memory = False

escribir_en_memoria = False

ga1 = ga.GeneticAlgorithm(datos, Y, functions, atributos, 2, memory, True, 15, 0.05, 25000)
find = ga1.start()
best_tree = ga1.getBestTree()

json_tree = best_tree.getJSON(best_tree.getRoot(), {}, 0)
print('')
print('')
j = dumps(json_tree, indent=2)
print(j)


with open('json_eq.txt', 'w') as f:
    f.write(j)

if escribir_en_memoria:
    ga1.saveTreeInMemory(best_tree)
    ga1.saveMemoryInFile('./memory.pkl')
