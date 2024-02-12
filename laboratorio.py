import kernel
import Function
import Tree
import pickle as pkl
import GeneticAlgorithm as ga
import pandas as pd
import numpy as np
import math
import csv_experiment
from json_tricks import dumps
import random


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

functions = [f_plus, f_div2, f_add, f_pow, f_dos, f_tres, f_cuatro, f_sen]

datos = []
for _ in range(150):
    datos.append([int(random.randrange(1, 15, 1))])

datos = np.asarray(datos)

y = [(x*x)/5*x for x in datos]


try:
    fileObject2 = open('./memory.pkl', 'rb')
    memory = pkl.load(fileObject2)
    fileObject2.close()
    use_memory = True
except:
    memory = []
    use_memory = False

ga1 = ga.GeneticAlgorithm(datos, y, functions, ['x'], 2, memory, True, 15, 0.05, 25000)
find = ga1.start()
best_tree = ga1.getBestTree()

json_tree = best_tree.getJSON(best_tree.getRoot(), {}, 0)
print('')
print('')
j = dumps(json_tree, indent=2)
print(j)

ga1.saveTreeInMemory(best_tree)
ga1.saveMemoryInFile('./memory.pkl')
