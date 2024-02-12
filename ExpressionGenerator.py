import random



def generate_expression(length):
    OPERATORS = ['+', '-', '*', '/','Q']
    MIN_RANGE = 1
    MAX_RANGE = 10

    stack = []
    stack = [(str(random.randint(MIN_RANGE, MAX_RANGE)))]
    for _ in range(length):       
        if random.random() > 0.25 and len(stack)>1:
            operator = random.choice(OPERATORS)
            right = stack.pop()
            left = stack.pop()
            stack.append(f"{left} {right} {operator}")
        else:
            stack.append(str(random.randint(MIN_RANGE, MAX_RANGE)))
    return stack.pop()



def random_suffix_expression(max_depth):
    # Lista de operadores y su aridad
    operators = {"+": 2, "-": 2, "*": 2, "/": 2, "L": 2, "Q":2}
    # Si la profundidad máxima ha sido alcanzada, generar un número aleatorio
    if max_depth == 0:
        return str(random.randint(1, 10))
    
    # Elegir un operador aleatorio y obtener su aridad
    operator, arity = random.choice(list(operators.items()))
    
    # Generar aleatoriamente los operandos
    operands = [random_suffix_expression(max_depth - 1) for i in range(arity)]
    
    # Unir los operandos y el operador en una expresión sufijo
    suffix_expression = " ".join(operands + [operator])
    
    return suffix_expression

