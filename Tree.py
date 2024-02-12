import node
import Function
import kernel
import random
import copy
import json
from pptree import *

class Tree:
    def __init__(self, functions, variables, depth = 15) -> None:
        self.root = None #Nodo principal
        self.depth = depth       
        self.functions = functions
        self.variables = variables
    
    def getRoot(self):
        return self.root
    
    def setVariables(self, var):
        self.variables = copy.deepcopy(var)
    
    def getVariables(self):
        return self.variables

    def getSubTree(self, cadena, pos):
        n_ant2 = None
        elem = cadena[pos]
        while not elem == ')' and pos < len(cadena):
            if not elem == '*' and not elem == '/':
                if not elem == '(':
                    n = node.node(elem, 'variable')
                    if not n_ant2 == None:
                        if n_ant2.getType() == 'function':
                            n_ant2.setChild(n)
                        else:
                            n_ant2 = copy.deepcopy(n)
                    else:
                        n_ant2 = copy.deepcopy(n)
            else:
                #Es una funcion
                if elem == '*':
                    nf = node.node(Function.function('*', 2, kernel.plus), 'function')
                if elem == '/':
                    nf = node.node(Function.function('/', 2, kernel.protected_div), 'function')
                nf.setChild(n_ant2)
                n_ant2 = copy.deepcopy(nf)
            
            pos += 1
            elem = cadena[pos]
        
        return n_ant2, pos

    def string2tree(self, cadena):
        import numpy as np

        sin_espacios = []
        for string in cadena:
            if string != ' ':
                sin_espacios.append(string.strip(' '))

        cadena = np.asarray(sin_espacios)
        print('')
        print(cadena)
        pos = 0
        n_ant = None
        while pos < len(cadena):
            if n_ant is None:
                left, pos = self.getSubTree(cadena, pos)
                
                pos += 1
                elem = cadena[pos]

                if elem == '*':
                    nf = node.node(Function.function('*', 2, kernel.plus), 'function')
                if elem == '/':
                    nf = node.node(Function.function('/', 2, kernel.protected_div), 'function')

                rigth, pos = self.getSubTree(cadena, pos+1)
                nf.setChild(left)            
                nf.setChild(rigth)

                n_ant = copy.deepcopy(nf)
            else:
                pos += 1
                elem = cadena[pos]
               
                if elem == '*':
                    nf = node.node(Function.function('*', 2, kernel.plus), 'function')
                if elem == '/':
                    nf = node.node(Function.function('/', 2, kernel.protected_div), 'function')
                
                nf.setChild(n_ant)
                rigth, pos = self.getSubTree(cadena, pos+1)
                nf.setChild(rigth)
                n_ant = copy.deepcopy(nf)
            
            pos += 1

        '''n_ant = None
        for elem in cadena:          
            if not elem == '*' and not elem == '/':
                if not elem == '(' and not elem == ')':
                    #Es una variable                   
                    n = node.node(elem, 'variable')
                    if not n_ant == None:
                        if n_ant.getType() == 'function':
                            n_ant.setChild(n)
                        else:
                            n_ant = n
                    else:
                        n_ant = n
            else:
                #Es una funcion
                if elem == '*':
                    nf = node.node(Function.function('*', 2, kernel.plus), 'function')
                if elem == '/':
                    nf = node.node(Function.function('/', 2, kernel.protected_div), 'function')
                nf.setChild(n_ant)
                n_ant = nf'''
        
        self.root = copy.deepcopy(n_ant)
      
    def show(self, nodo, s):
        if not nodo is None:
            if nodo.getType() == 'function':
                for elem in nodo.getChilds():
                    s += self.show(elem, s)
                s += nodo.getObject().getElem()
                return s
            else:
                s += nodo.getObject()
                return s
    
    def getJSON(self, nodo, json_tree, num):
        if not nodo is None:
            if nodo.getType() == 'function':                
                #Por cada hijo devolvemos los hijos
                hijos = []
                for elem in nodo.getChilds():                    
                    hijos.append(self.getJSON(elem, json_tree, num+1))

                return ({"operacion" + str(num):nodo.getObject().getElem(), "hijos" + str(num):hijos})                

            else:
                return {nodo.getObject()}
        else:
            return None

        
    def getNumTerminals(self, nodo, cantidad):
        #Devuelve el numero de parametros libres
        if nodo is None:
            return cantidad
        else:
            if not nodo.getType() == 'function':
                return cantidad + 1
            else:
                if nodo.getObject().getArity() == 0:
                    return cantidad + 1
                else:
                    for elem in nodo.getChilds():
                        cantidad += self.getNumTerminals(elem, cantidad)
                    
                    return cantidad
                
    def getNumFunctions(self, nodo, cantidad):
        #Devuelve el numero de parametros libres
        if nodo is None:
            return cantidad
        else:
            if nodo.getType() == 'function':
                if nodo.getObject().getArity() == 0:
                    return cantidad
                else:
                    cantidad += 1
                    for elem in nodo.getChilds():
                        cantidad += self.getNumTerminals(elem, cantidad)
                    return cantidad
            else:
              return cantidad
    
    def transform(self, variables, nodo):
        #Funcion para transformar las variables de un arbol por las actuales
        if not nodo.getType() == 'function':
            #Es una variable
            indice = random.randint(0, len(variables)-1)
            nodo.setObject(variables[indice])
        else:
            for elem in nodo.getChilds():
                self.transform(variables, elem)                   

    def evaluate(self, nodo, dictionary):
        try:
            #In dictionary we have the terminal values        
            if not nodo.getType() == 'function':
                return dictionary[nodo.getObject()]
            
            if nodo is not None:
                params = []
                for elem in nodo.getChilds():   
                    if not elem.getType() == 'function':                
                        param = dictionary[elem.getObject()]
                        params.append(param)
                    else:
                        result = self.evaluate(elem, dictionary)                    
                        params.append(result)
                #Ejecutamos el resultado
                
                result = nodo.exec(params)
                return result[0]
            else:
                return 0
        except:
            return None

    def init_random(self, level):
        #Inicializa el arbol de forma aleatoria
        #Puede que algunas expresiones no tenga sentido

        if level < self.depth:
            if random.random() > 0.5:
                #Creamos un nodo de tipo funcion
                indice = random.randint(0, len(self.functions)-1)

                if self.root is None:                    
                    self.root = node.node(self.functions[indice],'function')
                    for _ in range(self.functions[indice].getArity()):
                        self.root.setChild(self.init_random(level + 1))
                else:
                    n = node.node(self.functions[indice],'function')
                    for _ in range(self.functions[indice].getArity()):
                        n.setChild(self.init_random(level + 1))
                    
                    return n
            else:
                #Creamos un nodo de tipo variable
                indice = random.randint(0, len(self.variables)-1)
                if self.root is None:
                    self.root = node.node(self.variables[indice],'var')
                else:
                    return node.node(self.variables[indice],'var')
        else:
            indice = random.randint(0, len(self.variables)-1)
            return node.node(self.variables[indice],'var')
    
    def getFunctions(self, arity):
        result = []
        for f in self.functions:
            if f.getArity() == arity:
                result.append(f)
        
        return result

    def crossover(self, tree2):
        nodo1 = self.root
        nodo2 = tree2.root

        hijo1 = nodo1.getRandomChild()
        hijo2 = nodo2.getRandomChild()
        
        if not len(nodo1.childs) == 0:
            nodo1.childs[nodo1.childs.index(hijo1)] = hijo2 
        else:
            nodo1.childs.append(hijo2)

        if not len(nodo2.childs) == 0:            
            nodo2.childs[nodo2.childs.index(hijo2)] = hijo1       
        else:
            nodo2.childs.append(hijo1) 

        arbol1 = Tree(self.functions, self.variables)
        arbol1.root = copy.deepcopy(nodo1)
        arbol2 = Tree(self.functions, self.variables)
        arbol2.root = copy.deepcopy(nodo2)

        return arbol1, arbol2

    def mutation(self, nodo, taxe = 0.25, do=True):
        #Metodo que muta un nodo del arbol
        if not isinstance(nodo, list):
            if do:
                if random.random() > taxe:
                    if nodo.getType() == 'function':
                        fun = self.getFunctions(nodo.getObject().getArity())
                        if len(fun) > 0:                        
                            indice = random.randint(0, len(fun)-1)
                            nodo.setObject(fun[indice])
                            for c in nodo.getChilds():
                                self.mutation(c, taxe, False)
                    else:                       
                        indice = random.randint(0, len(self.variables)-1)
                        nodo.setObject(self.variables[indice])
                        for c in nodo.getChilds():
                            self.mutation(c, taxe, False)
                else:
                    if not isinstance(nodo, list):
                        for c in nodo.getChilds():
                            self.mutation(c, taxe, do)                

    def displayTree(self):
        if self.root.getType() == 'function':
            root = self.root.getObject().getElem()   
            n_root = Node(root)
        else:
            n_root = Node(self.root.getObject())        
        
        for elem in self.root.getChilds():    
            n = self.displayNode(elem, n_root)          
        
        print_tree(n_root, horizontal=False)
    
    def displayNode(self, node2, parent):
        if node2 is None:
            return []
        
        if not node2 == [] and not node2 is None: 
            if node2.getType() == 'function':          
                n = Node(node2.getObject().getElem(), parent)
            else:
                n = Node(node2.getObject(), parent)
        
            for elem in node2.getChilds():
                self.displayNode(elem, n)

            return []
              
        else:
            return []     