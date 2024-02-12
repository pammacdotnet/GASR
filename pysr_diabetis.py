import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import Function
import GeneticAlgorithm as ga
import numpy as np
import kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import random
import kernel
import pysr
from pysr import PySRRegressor



from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

#Web ==> https://as595.github.io/classification/
#Dataset ==> conjunto de datos HTRU2
#Paper ==> https://figshare.com/articles/dataset/HTRU2/3080389/1

df = pd.read_csv('H:/Induccion fisica/diabetis/diabetes_prediction_dataset.csv') #pd.read_csv('./estrellas/pulsar.csv')



print('')
print ('Dataset has %d rows and %d columns including features and labels'%(df.shape[0],df.shape[1]))

etiquetas = df['diabetes']

features = df.drop(['diabetes','gender','age','smoking_history'], axis=1)

print('')
print(features.columns)

list_terminals = features.columns.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.90, random_state=45)


X_train = X_train.to_numpy() #X_smote.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

y_train = y_train.to_numpy()

print('')
print('X_train : ', X_train.shape)
print('X_test : ', X_test.shape)
print('')

objective = """
#import Pkg;
#Pkg.add("Metrics");

#using Metrics

#import Pkg
#Pkg.add("CategoricalArrays")


function kappa_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
  (prediction, completion) = eval_tree_array(tree, dataset.X, options)
    if !completion
        return L(Inf)
    end
    
    pred = Vector{Int64}()
    for e in prediction
        p = round(1.0 ./ (1.0 + exp(-e)))        
        push!(pred, trunc(Int,p))
    end

    real = Vector{Int64}()
    for e in dataset.y
        push!(real, trunc(Int,e))
    end

    acc = 0
    
    for i in 1:length(real)
        if real[i] == pred[i]
            acc = acc + 1
        end
    end
    
    return (acc/length(real))*100;
end
"""

model = PySRRegressor(
    niterations=1000,
    populations=15,   
    binary_operators=["+", "-", "/", "*"],
    unary_operators=["sqrt", "exp", "log"],
    full_objective = objective,
)

model.fit(X_train, y_train, variable_names=["hypertension","heart_disease","bmi","HbA1c_level","blood_glucose"])

print('')
print(model)

print(model.sympy())
print('')
print(model.latex())
