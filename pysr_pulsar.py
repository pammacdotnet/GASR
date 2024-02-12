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

df = pd.read_csv('H:/Induccion fisica/articulo/Revista española de fisica/Datos pulsares/HTRU_2.csv') #pd.read_csv('./estrellas/pulsar.csv')


print(df.columns)
print('')
print ('Dataset has %d rows and %d columns including features and labels'%(df.shape[0],df.shape[1]))

etiquetas = df['class']

features = df.drop('class', axis=1)

list_terminals = features.columns.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(features, etiquetas, test_size=0.97, random_state=45)


X_train = X_train.to_numpy() #X_smote.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

y_train = y_train.to_numpy()

#PySR

#Parametros de configuración
'''default_pysr_params = dict(
    populations=15,
    model_selection="best",
)'''

#Funcion de evaluacion redefinida

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

loss_function = """
    function binary_cross_entropy_loss(y_pred, y_true)
        epsilon = 1e-10
        num_samples = length(y_true)
        loss = 0.0
        
        for i in 1:num_samples
            # Avoid division by zero
            y_pred[i] = max(min(y_pred[i], 1.0 - epsilon), epsilon)
            
            # Calculate the loss for each sample
            loss -= y_true[i] * log(y_pred[i]) + (1 - y_true[i]) * log(1 - y_pred[i])
        end
        
        # Average the loss across all samples
        loss /= num_samples
        
        return loss
    end
"""
'''
classes = union(pred, real)
    num_classes = length(classes)
    confusion_matrix = zeros(Int, num_classes, num_classes)

    for i in 1:length(real)
        true_label = real[i]
        pred_label = pred[i]
        true_index = findfirst(classes .== true_label)
        pred_index = findfirst(classes .== pred_label)
        confusion_matrix[true_index, pred_index] += 1
    end

    total_samples = sum(confusion_matrix)
    expected_matrix = zeros(Float64, num_classes, num_classes)
    
    for i in 1:num_classes
        row_sum = sum(confusion_matrix[i, :])
        col_sum = sum(confusion_matrix[:, i])
    
        for j in 1:num_classes
            expected_matrix[i, j] = (row_sum * col_sum) / total_samples
        end
    end

    numerator = sum((confusion_matrix .- expected_matrix) .^ 2)
    denominator = sum((total_samples * expected_matrix) .^ 2)

    kappa = 1.0 - (numerator / denominator)s
'''

model = PySRRegressor(
    niterations=100,
    populations=10,   
    binary_operators=["+", "-", "/"],
    unary_operators=["sqrt", "exp", "log"],
    full_objective = objective,
)

model.fit(X_train, y_train, variable_names=["mean_int_pf","std_pf","ex_kurt_pf","skew_pf","mean_dm","std_dm","kurt_dm","skew_dm"])

print('')
print(model)

print(model.sympy())
print('')
print(model.latex())
