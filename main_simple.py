import kernel
import Function
import Tree
import pickle as pkl
import GeneticAlgorithm as ga
import pandas as pd
import numpy as np
import math

def substituir_dimensiones(cadena, variables_fisicas): 
    for dim, var in variables_fisicas.items():
        cadena = cadena.replace(dim, f"({str(var)})")
    
    return cadena

def getUnits(unidades, variable):
    result = ''
    reg = (unidades[unidades['Variable']==variable]).to_numpy()[0]

    if not reg[2] == 0:
        if not reg[2] == 1:
            for _ in range(abs(int(reg[2]))):
                result += 'M'
        else:
            result += 'M'

    if not reg[3] == 0:
        if not reg[3] == 1:
            for _ in range(abs(int(reg[3]))):
                result += 'S'
        else:
            result += 'S'
    
    if not reg[4] == 0:
        if not reg[4] == 1:
            for _ in range(abs(int(reg[4]))):
                result += 'KG'
        else:
            result += 'KG'
    
    if not reg[5] == 0:
        if not reg[5] == 1:
            for _ in range(abs(int(reg[5]))):
                result += 'T'
        else:
            result += 'T'
    
    if not reg[6] == 0:
        if not reg[6] == 1:
            for _ in range(abs(int(reg[6]))):
                result += 'V'
        else:
            result += 'V'
    
    return result


'''file = 'I.34.8'

df = pd.read_csv('H:/Pryecto Fisica UNIR/AI Feynmann/Feynman_with_units/' + file, sep=' ', header=None)
units = pd.read_csv('H:/Pryecto Fisica UNIR/AI Feynmann/FeynmanEquations.csv', sep=',')
var = units[units['Filename'] == file].to_numpy()[0]
list_terminals = []

pos = 5
cantidad = 0
for _ in range(int(var[4])+1):
    if isinstance(var[pos], str):
        list_terminals.append((str(var[pos])))
        cantidad += 1
    pos += 3
    
salida = units[units['Filename'] == file]['Output'].to_numpy()[0]

goal = df.iloc[0:100,cantidad]
goal = goal.to_numpy()
df.pop(df.columns[cantidad])
datos = df.to_numpy()

#Para cada variable debemos cargar las unidades
unidades = pd.read_csv('H:/Pryecto Fisica UNIR/AI Feynmann/units.csv', sep=',')
diccionario_unidades = {}
for var in list_terminals:   
    r = getUnits(unidades, var) 
    diccionario_unidades.update({r:var})'''

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

#functions = [f_add, f_plus, f_div2, f_uno, f_dos, f_tres, f_cuatro, f_sen, f_cos, f_sqrt, f_e, f_pi, f_pow, f_log]
functions = [f_plus, f_sub, f_add]

file = 'I.11.19'

df = pd.read_csv('H:/Pryecto Fisica UNIR/AI Feynmann/Feynman_with_units/' + file, sep=' ', header=None)
units = pd.read_csv('H:/Pryecto Fisica UNIR/AI Feynmann/FeynmanEquations.csv', sep=',')
var = units[units['Filename'] == file].to_numpy()[0]
list_terminals = []

pos = 5
cantidad = 0
for _ in range(int(var[4])+1):
    if isinstance(var[pos], str):
        list_terminals.append((str(var[pos])))
        cantidad += 1
    pos += 3
    
salida = units[units['Filename'] == file]['Output'].to_numpy()[0]

goal = df.iloc[0:100,cantidad]
goal = goal.to_numpy()


df.pop(df.columns[cantidad])
datos = df.to_numpy()
datos1 = datos[0:100,:]

'''list_terminals = ['Longitud','Masa','Gravedad','Angulo']

import random

q1 = [[random.random()] for x in range(1, 250)]
q2 = [[(random.random())] for x in range(1, 250)]
d = [[9.81] for x in range(1, 250)]
k = [[i/6] for i in range(1, 250)]

datosC = []
datosC.append(q1)
datosC.append(q2)
datosC.append(d)
datosC.append(k)
datosC = np.asarray(datosC)

datosC = np.reshape(datosC, (datosC.shape[1], datosC.shape[0]))


goal = [(2*math.pi)*math.sqrt(elem[0]/elem[2]) for elem in datosC]'''

#Cargamos la memoria

try:
    fileObject2 = open('./memory.pkl', 'rb')
    memory = pkl.load(fileObject2)
    fileObject2.close()
    use_memory = True
except:
    memory = []
    use_memory = False


ga1 = ga.GeneticAlgorithm(datos1, goal, functions, list_terminals, 2, memory, True, 15, 0.05, 25000)
find = ga1.start()
best_tree = ga1.getBestTree()
ga1.saveTreeInMemory(best_tree)

ga1.saveMemoryInFile('./memory.pkl')


