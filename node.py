import copy
import random

class node:
    def __init__(self, object, type) -> None:
        self.value = copy.deepcopy(object)
        self.type = type
        self.childs = []
    
    def getObject(self):
        return self.value

    def numChilds(self):
        return len(self.childs)
    
    def getChild(self, pos):
        return self.childs[pos]

    def getChilds(self):
        return self.childs

    def getRandomChild(self):
        if self.childs:
            return random.choice(self.childs)
        else:
            return []

    def setChild(self, child):
        self.childs.append(child)
    
    def setObject(self, new_object):
        self.value = copy.deepcopy(new_object)
    
    def getType(self):
        return self.type
    
    def exec(self, params):    
        if self.type == 'function':   
            return self.value.exec(params)