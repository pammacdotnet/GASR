from keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Attention
from keras.models import Model, Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Definir una función para calcular la aproximación de Taylor
def taylor_approximation(model, x):
    # Crear un objeto GradientTape para registrar las operaciones en el grafo
    with tf.GradientTape() as tape1:
        # Crear otro objeto GradientTape dentro del primero para registrar las operaciones adicionales
        with tf.GradientTape() as tape2:
            # Obtener la salida de la red neuronal
            y_pred = model(x)
        # Obtener los gradientes de la salida con respecto a la entrada
        gradients = tape2.gradient(y_pred, x)
    # Obtener la hessiana de la salida con respecto a la entrada
    hessians = tape1.gradient(gradients, x)
    # Calcular la aproximación de Taylor de la red neuronal
    return y_pred, gradients, hessians

# Definimos la arquitectura de la red neuronal
file = 'I.9.18'

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
goal_test = df.iloc[200:300, cantidad]
goal = goal.to_numpy()
goal_test = goal_test.to_numpy()

df.pop(df.columns[cantidad])
datos = df.iloc[0:100, 0:len(list_terminals)]
datos_test = df.iloc[200:300, 0:len(list_terminals)]
datos = datos.to_numpy()
datos_test = datos_test.to_numpy()

#************ CONSTRUIMOS UN REGRESOR NUMERICO CON UNA RED NEURONAL *************************************


def obtener_coeficientes_de_polinomio(modelo):
    """
    Función para obtener los coeficientes del polinomio que aproxima la función aprendida por una red neuronal.
    
    Argumentos:
    modelo -- objeto modelo de Keras
    
    Retorna:
    coeficientes -- lista de coeficientes del polinomio, donde el índice i corresponde al coeficiente de x^i
    """
    # Obtener los pesos de la red neuronal
    pesos = modelo.get_weights()
    
    # Inicializar la lista de coeficientes
    coeficientes = []
    
    # Obtener el número de capas
    num_capas = len(pesos) // 2
    
    # Obtener el número de neuronas en cada capa
    num_neuronas = [pesos[i * 2].shape[1] for i in range(num_capas)]
    
    # Obtener las funciones de activación de cada capa
    funciones_activacion = [capa.activation.__name__ for capa in modelo.layers]
    
    # Añadir el coeficiente constante
    coeficientes.append(pesos[1][0])
    
    # Añadir los coeficientes lineales
    for i in range(num_neuronas[0]):
        coeficientes.append(pesos[0][0][i])
    
    # Calcular los coeficientes para el resto de grados
    for grado in range(2, max(num_neuronas) + 1):
        # Inicializar el coeficiente para el grado actual
        coeficiente_actual = 0
        
        # Iterar sobre todas las neuronas de todas las capas
        for i in range(num_capas):
            for j in range(num_neuronas[i]):
                # Obtener el peso correspondiente
                peso = pesos[i * 2].T[j][0]
                
                # Obtener la función de activación correspondiente
                funcion_activacion = funciones_activacion[i + 1]
                
                # Calcular el coeficiente para el grado actual
                if funcion_activacion == 'relu':
                    coeficiente_actual += max(0, coeficientes[1 + j]) * peso
                elif funcion_activacion == 'sigmoid':
                    coeficiente_actual += 1 / (1 + np.exp(-coeficientes[1 + j])) * peso
                elif funcion_activacion == 'tanh':
                    coeficiente_actual += np.tanh(coeficientes[1 + j]) * peso
                else:
                    raise ValueError(f"Función de activación desconocida: {funcion_activacion}")
        
        # Añadir el coeficiente para el grado actual
        coeficientes.append(coeficiente_actual)
    
    return coeficientes



X_train = np.asarray([x for x in range(1, 450)]).reshape(-1,1)
y_train = np.asarray([x^3 for x in range(1, 450)]).reshape(-1,1)

# Normalizamos los datos de entrada
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)




model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
print(model.summary())

model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=1)


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('')
print(obtener_coeficientes_de_polinomio(model))

