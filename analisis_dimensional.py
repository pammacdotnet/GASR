import numpy as np
from json_tricks import dumps
import unidades
import Tree
import kernel
import Function
import GeneticAlgorithm as ga
import csv_experiment


def separarElementos(cadena):
    pos = 0
    variable = ""
    resultado = []

    while pos < len(cadena):
        elem = cadena[pos]        
        if elem == '(':
            resultado.append(elem)
        elif elem == ')':
            if not variable == "":
                resultado.append(variable)
                variable = ""
            resultado.append(elem)
        elif elem == '*':
            if not variable == "":
                resultado.append(variable)
                variable = ""            
            resultado.append(elem)
            variable = ""
        elif elem == '/':
            if not variable == "":
                resultado.append(variable)
                variable = ""           
            resultado.append(elem)
            variable = ""
        else:
            variable += elem
        
        pos += 1
    
    return np.asarray(resultado)
        

def eliminar_espacios(cadena):
    resultado = ''
    for elem in cadena:
        if not elem == ' ':
            resultado += elem
    
    return resultado

variables = {'G':'m*m*m/kg*s*s','M1':'kg','R':'m','M2':'kg'} #{'M':'kg','G':'(m)/(s*s)','A':'m'} #{'G':'m*m*m/kg*s*s','M1':'kg','R':'m'} #{'K':'kg/s*s','DIST':'m'} #{'G':'m*m*m/kg*s*s','M1':'kg','R':'m'}

fuerza = "( kg * m ) / ( s * s )"
var = "( kg * kg ) / ( m * m )" #"( m ) / ( s * s )" #"( kg * kg ) / ( m * m )"

file_csv = './experimentos/gravitacion_universal.csv'

u1 = unidades.unidades(fuerza, var)

x = u1.resolver()


var = eliminar_espacios(var)

resultado = "(" + var + ")" + '*' + "(" + x + ")"

print(resultado)

#Substituimos en la ecuacion
for variable, unidad in variables.items():
    resultado = resultado.replace(unidad, variable)

print('')
print(resultado)
primera_eq = separarElementos(resultado)

print('')
print(primera_eq)



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

index_goal = 0

datos = csv_experiment.get_data_experiment(file_csv)
atributos = csv_experiment.get_atributes(file_csv)

print('')
at = atributos.copy()
print('Valor objetivo : ', atributos[index_goal])


atributos = np.delete(atributos, index_goal).to_numpy()
atributos = [x.upper() for x in atributos]

#Aplicamos un factor de escala

Y = datos[:,index_goal]

datos = np.delete(datos, (index_goal), axis=1)

'''atributos = ['G','M1','M2','R']

t1 = Tree.Tree(functions, atributos)
t1.string2tree(primera_eq)
print('')
t1.displayTree()

import random

cantidad = 100

masa = [random.randint(2, 25) for i in range(cantidad)]
masa2 = [random.randint(2, 25) for i in range(cantidad)]
R = [random.random()*5 for i in range(cantidad)]
Y = [10*i*j for i,j in zip(masa1, R)]

datos = []
for i, j in zip(masa, altura):
    datos.append([i, j, 10])

datos = np.asarray(datos)'''

t1 = Tree.Tree(functions, atributos)
t1.string2tree(primera_eq)
print('')
t1.displayTree()

input('>>')

ga1 = ga.GeneticAlgorithm(datos, Y, functions, atributos, 2, False, [], False, 2, 0.25, 1500)
ga1.setChromosome(t1)

f = ga1.fitness(t1)
print('')
print('Fitness primera eq. : ', f)


if f > 1e-7:
    find = ga1.start()

best_tree = ga1.getBestTree()