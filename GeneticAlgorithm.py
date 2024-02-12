import Tree
import copy
import random
import pickle
import os
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import numpy as np
import kernel
import matplotlib.pyplot as plt



class GeneticAlgorithm:
    def __init__(self, data, goal, functions, variables, num_best, random_population=False, memory=[],
                use_memory = False, num_population=10, mutation_taxe=0.15,
                epochs=90000, verbose=True, clasification=False, activation_function='tanh',
                kappa_finished=0.75, reinicio=100) -> None:
        self.population = []
        self.num_population = num_population
        self.epochs = epochs
        self.num_best = num_best

        self.data = data
        self.goal = goal
        self.mutation_taxe = mutation_taxe

        self.functions = functions
        self.variables = variables

        self.array_fitness = []
        self.array_tree = []

        self.memory_fitness = []
        self.array_memory_tree = []

        self.clasification = clasification

        if self.clasification:
            self.best_fitness = 0
        else:
            self.best_fitness = 100000000000

        self.best_tree = None
        self.verbose = verbose

        self.rand_population = random_population

        #Guarda informacion de expresiones buenas que ha aprendido anteriormente
        self.use_memory = use_memory
        self.memory = memory        

        self.activation_function = activation_function
        self.kappa_finished = kappa_finished
        self.restart = reinicio
    
    def sigmoid(self, x):
        import math
        try:
            return 1 / (1 + math.exp(-x))
        except:
            return 1000

    def init(self, pop):
        self.population = pop

    def init_random(self):
        #Inicializamos de forma aleatoria la poblacion
        try:
            for _ in range(self.num_population):
                t = Tree.Tree(self.functions, self.variables)
                t.init_random(0)
                self.population.append(copy.deepcopy(t))
            if self.verbose:
                print('[+]- Poblacion creada : ok')
        except:
            print('[-]- Error al crear la poblacion')


    def fitness(self, tree):   
        f = 0               
        for i in range(0, len(self.goal)):
            d = self.data[i]            
            g = self.goal[i]
            dict = {}
            for j in range(0, len(self.variables)):
                name = self.variables[j]
                value = d[j]
                dict.update({name:value})
            
            result = tree.evaluate(tree.getRoot(), dict)  
            if not result is None:                           
                    f += ((result-g)*(result-g))
            else:
                f += 10000
        
        if (f/len(self.goal)) < 0:
            f1 = -(f/len(self.goal))
        else:
            f1 = (f/len(self.goal))       

        if f1 > 1000000000:
            return 1000000
        else:
            return f1   

    def fitness_clasification(self, tree):   
        f = 0        
        r = []       
        for i in range(0, len(self.goal)):
            d = self.data[i]            
            g = self.goal[i]
            dict = {}
            for j in range(0, len(self.variables)):
                name = self.variables[j]
                value = d[j]
                dict.update({name:value})
            
            result = tree.evaluate(tree.getRoot(), dict)  
            r.append(result)
            '''if not result is None:   
                try:                        
                    if round(self.sigmoid(result)) == g:
                        f += 1
                except:
                    f += -10
            else:
                f += -10000'''
        r = np.asarray(r)
        y_pred = []
        for elem in r:
            try:
                #Ojo con esto !!!
                if not elem is None:
                    if self.activation_function == 'sigmoid':
                        y_pred.append(round(kernel.sigmoid(elem)))
                    elif self.activation_function == 'tanh':
                        y_pred.append(round(kernel.tanh(elem)))
                    elif self.activation_function == 'leaky_relu':
                        y_pred.append(round(kernel.leaky_relu(elem)))


                else:
                    y_pred.append(random.randint(0,2))
            except:
                y_pred.append(random.randint(0,2))
        
        y_pred = np.asarray(y_pred)
        return cohen_kappa_score(y_pred, self.goal)
    
    def bubbleSort(self):
        n = len(self.array_fitness)
        for i in range(n-1):
            for j in range(n-1-i):
                if self.clasification:
                    if self.array_fitness[j] < self.array_fitness[j+1]:
                        self.array_fitness[j], self.array_fitness[j+1] = self.array_fitness[j+1], self.array_fitness[j]
                        self.array_tree[j], self.array_tree[j+1] = self.array_tree[j+1], self.array_tree[j]
                else:
                    if self.array_fitness[j] > self.array_fitness[j+1]:
                        self.array_fitness[j], self.array_fitness[j+1] = self.array_fitness[j+1], self.array_fitness[j]
                        self.array_tree[j], self.array_tree[j+1] = self.array_tree[j+1], self.array_tree[j]
    
    def bubbleSortMemory(self):
        n = len(self.memory_fitness)
        for i in range(n-1):
            for j in range(n-1-i):
                if self.memory_fitness[j] > self.memory_fitness[j+1]:
                    self.memory_fitness[j], self.memory_fitness[j+1] = self.memory_fitness[j+1], self.memory_fitness[j]
                    self.array_memory_tree[j], self.array_memory_tree[j+1] = self.array_memory_tree[j+1], self.array_memory_tree[j]


    def saveTreeInMemory(self, tree):
        self.memory.append(tree)
    
    def saveMemoryInFile(self, file):
        fileObject = open(file, 'wb')
        pickle.dump(self.memory, fileObject)

    def setChromosome(self, c):
        self.population.append(c)

    def start(self):
        hist = []
        fitness_anterior = 0
        fitness_actual = 0
        num_igual = 0

        if self.verbose:
            print('[+]- Start GA')

        iteration = 0
        it = 0
        find = False

        #Creamos la poblacion inicial de forma aleatoria
        if self.rand_population:
            self.init_random()
        while(iteration < self.epochs):
            #Inicializamos las variables a 0
            new_population = []
            self.array_fitness = []
            self.array_tree = []
            mejores = []
            self.memory_fitness = []
            self.array_memory_tree = []

            if self.verbose:
                os.system('cls')

            if self.verbose:
                print('Iteracions : ', (iteration + 1))
            
            #Calculamos el valor de fitness de cada elemento
            for elem in self.population:   
                if self.clasification:
                    f = self.fitness_clasification(elem)
                else:
                    f = self.fitness(elem)                
                self.array_fitness.append(f)
                self.array_tree.append(copy.deepcopy(elem))
                
            #Ordenamos la poblacion de menor fitness a mayor fitness
            self.bubbleSort() 
           
            #Ecogemos los N mejores individuos
            mejores = self.array_tree[0:self.num_best]

            #Uso de memoria genetica
            if self.use_memory:
                #Calculamos el fitnes para los elementos de la memoria
                if self.verbose:
                    print('[+]- Fitness memory')

                for elem in self.memory:                
                    cloneTree = copy.deepcopy(elem)
                    cloneTree.transform(self.variables, cloneTree.getRoot())                       
                    cloneTree.setVariables(self.variables)                       
                    f = self.fitness(cloneTree)                  
                    self.memory_fitness.append(f)
                    self.array_memory_tree.append(copy.deepcopy(cloneTree))
                
                self.bubbleSortMemory()

                if len(self.array_memory_tree) > 0:
                    new_population.append(copy.deepcopy(self.array_memory_tree[0]))


            if self.verbose:
                print('     -> mejor fitness iteracion : ', self.array_fitness[0])
                print('')

            hist.append(self.array_fitness[0])

            #Observamos de esta generacion si existe algun elemento mejor
            #que el mejor global
            if self.clasification:
                if self.best_fitness < self.array_fitness[0]:
                    self.best_fitness = self.array_fitness[0]               
                    num_igual = 0
                    self.best_tree = copy.deepcopy(mejores[0])  
                else:                
                    num_igual += 1
            else:
                if self.best_fitness > self.array_fitness[0]:
                    self.best_fitness = self.array_fitness[0]               
                    num_igual = 0
                    self.best_tree = copy.deepcopy(mejores[0])  
                else:                
                    num_igual += 1
            
            if self.verbose:
                print('     -> Best_fitness :', self.best_fitness)
                print('     -> Best Tree :')
                print('')

            if self.verbose:
                if not self.best_tree is None:
                    self.best_tree.displayTree()
            
            #Añadimos los N mejores a la nueva poblacion para la siguiente
            #generacion
            for elem in mejores:
                new_population.append(copy.deepcopy(elem))  
            
            #Realizamos la operacion de Crossover con los 2 mejores
            tree1 = copy.deepcopy(mejores[0])
            '''tree2 = copy.deepcopy(mejores[1])

            hijo1, hijo2 = tree1.crossover(tree2)
            hijo3, hijo4 = tree2.crossover(tree1)

            #Añadimos los hijos del crossover a la nueva poblacion
            new_population.append(copy.deepcopy(hijo1))
            new_population.append(copy.deepcopy(hijo2))
            new_population.append(copy.deepcopy(hijo3))
            new_population.append(copy.deepcopy(hijo4))'''

            '''if self.use_memory:
                hijo3, hijo4 = self.array_memory_tree[0].crossover(tree1)

                new_population.append(copy.deepcopy(hijo3))
                new_population.append(copy.deepcopy(hijo4))'''

            '''hijo1, hijo2 = tree2.crossover(tree1)

            new_population.append(copy.deepcopy(hijo1))
            new_population.append(copy.deepcopy(hijo2))'''


            if num_igual > self.restart:
                #Reiniciamos todo para evitar caer en un minimo local
                new_population = []
                self.array_fitness = []
                self.array_tree = []
                mejores = []
                self.population = []
                num_igual = 0

                if self.clasification:
                    self.best_fitness = -100000
                else:
                    self.best_fitness = 100000

                self.best_tree = None
                self.init_random()
            else:
                #Creamos nuevos elementos aleatorios para completar
                #la cantidad de individuos en la nueva generacion

                r = self.num_population - len(new_population)
                for _ in range(0, r):
                    t = Tree.Tree(self.functions, self.variables)
                    t.init_random(0)                 
                    new_population.append(copy.deepcopy(t))            
                
                if not self.best_tree is None:
                    new_population.append(copy.deepcopy(self.best_tree))

                #Realizamos la mutacion de la nueva poblacion
                for elem in new_population:       
                    elem.mutation(elem.getRoot(), self.mutation_taxe, True)
                
                #new_population.append(self.best_tree)

                self.population = copy.deepcopy(new_population) 

            #Condicion de fin
            if self.clasification:
                if self.best_fitness > self.kappa_finished:
                    iteration = self.epochs
                    find = True
            else:
                if self.best_fitness < 2.5e-6:
                    iteration = self.epochs
                    find = True
            
            iteration += 1
            it += 1
        
        if self.verbose:
            os.system('cls')
            print('')
            print(' ***** Algoritmo Finalizado **********')
            print('')
            print('Mejor resultado : ', self.best_fitness)
            print('')
            print('Se ha tardado :', it + 1)

        if self.verbose:
            self.best_tree.displayTree()

        '''plt.plot(hist)
        plt.show()

        print(hist)'''

        return find  


    def getBestTree(self):
        return self.best_tree



    
