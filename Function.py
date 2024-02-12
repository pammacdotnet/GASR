from itertools import starmap 

class function:
    def __init__(self, symbol, arity, exec_function) -> None:
        self.symbol = symbol
        self.arity = arity
        self.exec_function = exec_function

    def getElem(self):
        return self.symbol
    
    def getSymbol(self):
        return self.symbol    
    
    def getArity(self):
        return self.arity

    def show(self):
        print(self.symbol)
    
    def info(self):
        print('Symbol: ', self.symbol)
        print('Arity: ', self.arity)
        print('Function: ', self.exec_function)
    
    def exec(self, params):       
        return list(starmap(self.exec_function, [params]))

    def isFunction(self):
        return True
