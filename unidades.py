import numpy as np
import math
import fractions
import sympy
from sympy.solvers import solve
from sympy import Symbol, Eq

class unidades:
    def __init__(self, unidades_objetivo, unidades_independiente) -> None:
        self.SI = {'kg':3,'m':5,'s':7,'K':11,'MOL':13,'CD':17,'A':19}

        self.target = unidades_objetivo
        self.independent = unidades_independiente
    

    def generar_notacion_polaca_inversa(self, expresion):
        precedencia = {'*': 3, '/': 3, '+': 2, '-': 2}
        salida = []
        pila = []
        for token in expresion.split():
            if token.isalnum():
                salida.append(token)
            elif token in precedencia:
                while (pila and precedencia[token] <= precedencia.get(pila[-1], 0)):
                    salida.append(pila.pop())
                pila.append(token)
            elif token == '(':
                pila.append(token)
            elif token == ')':
                while pila and pila[-1] != '(':
                    salida.append(pila.pop())
                pila.pop()
        while pila:
            salida.append(pila.pop())
        return ' '.join(salida)
    
    def evaluar_notacion_polaca(self, expresion):
        pila = []
        operadores = set(['+', '-', '*', '/'])
        for token in expresion.split():
            if token in operadores:
                operando2 = pila.pop()
                operando1 = pila.pop()
                resultado = eval(str(operando1) + token + str(operando2))
                pila.append(resultado)
            else:
                pila.append(float(token))
        return pila.pop()

    def notacion_polaca_a_fraccion(self, expresion):
        pila = []
        operadores = ["+", "-", "*", "/"]
        for token in expresion.split():
            if token not in operadores:
                pila.append(fractions.Fraction(token))
            else:
                b = pila.pop()
                a = pila.pop()
                if token == "+":
                    resultado = a + b
                elif token == "-":
                    resultado = a - b
                elif token == "*":
                    resultado = a * b
                else:
                    resultado = a / b
                pila.append(resultado)
        return pila.pop()

    def decimal_a_fraccion(self, decimal):
        n = len(str(decimal).split('.')[1])
        num = int(decimal * 10**n)
        den = 10**n
        mcd = math.gcd(num, den)
        return num//mcd, den//mcd

    def simplificar_fraccion(self, num, den):
        mcd = math.gcd(num, den)
        return num // mcd, den // mcd

    def factorizar_numero(self, numero):
        factores = sympy.factorint(numero)
        lista_factores = []
        for factor, potencia in factores.items():
            lista_factores.extend([factor] * potencia)
        return lista_factores

    def factorizar_numero(self, numero):
        factores = sympy.factorint(numero)
        lista_factores = []
        for factor, potencia in factores.items():
            lista_factores.extend([factor] * potencia)
        return lista_factores

    def transformar_a_SI(self, expresion):
        key_list = list(self.SI.keys())
        val_list = list(self.SI.values())   

        a = expresion.p
        b = expresion.q
        
        resultado = ''
        if a > 1:            
            #Debemos ir dividiendo hasta que no podamos dividir mas
            factores = self.factorizar_numero(a)
            primero = True
            for f in factores:
                position = val_list.index(f)
                if primero:
                    resultado += str(key_list[position])
                    primero = False
                else:
                    resultado += '*' + str(key_list[position])
        else:
            resultado += '1'
        
        if b > 1:
            resultado += '/'          
            #Debemos ir dividiendo hasta que no podamos dividir mas
            factores = self.factorizar_numero(b)
            primero = True
            for f in factores:
                position = val_list.index(f)
                if primero:
                    resultado += str(key_list[position])
                    primero = False
                else:
                    resultado += '*' + str(key_list[position])
        
        return resultado
    
    def convertir(self, expr):        
        lista_unidades = expr.split(' ')
        lista = []
        for elem in lista_unidades:
            try:
                lista.append(self.SI[elem])
            except:
                lista.append(elem)

        res = ''
        first = 0
        for elem in lista:
            if first == 0:
                res += str(elem)
                first = 1
            else:
                res += ' ' + str(elem)

        result = self.notacion_polaca_a_fraccion(res)

        
        return result

    def resolver(self):
        objetivo = self.generar_notacion_polaca_inversa(self.target)
        result = self.convertir(objetivo)
        ind = self.generar_notacion_polaca_inversa(self.independent)
        ind = self.convertir(ind)
            
        #Montamos la ecuacion
        x = Symbol('x')
        s = Eq((ind*x), result)
        s = solve(s)
        
        resultado = self.transformar_a_SI(s[0])

        return resultado
