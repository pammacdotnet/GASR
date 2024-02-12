import pandas as pd
import numpy as np


def get_data_experiment(file):
    '''
        This function open a csv file with experiment data
        the struct of the file is like as:
                      atribute 1, atribute 2, ..., atribute n
        observation 1
        observation 2
        observation n

        Return:
             - Dataframe
    '''

    data = pd.read_csv(file, delimiter=';', skiprows=[0])
    data = data.to_numpy()
    data = data.astype(float)

    return np.asarray(data)

def get_atributes(file):
    '''
        This function return the name of atributes
        
        Return:
            - Name of atributes
    '''
    data = pd.read_csv(file, delimiter=';')
    return data.columns

def delete_units(columns):
    '''
        This function rename columns
        the columns form are nam(units)
    '''
    atributes = []
    for c in columns:
        cn = c[:c.index("(")]
        atributes.append(cn)
    
    return np.asarray(atributes)
