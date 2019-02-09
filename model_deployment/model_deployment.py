#!/usr/bin/python
import numpy as np
from sklearn.externals import joblib

def predict_cluster(*args):
    model = joblib.load('model_deployment/iris_model_1.pkl')
    array_data = np.array(list(args[0].values())).reshape(1, -1)
    result = model.predict(array_data)[0]
    if result == 0:
        label = 'Iris-setosa'
    elif result == 1:
        label = 'Iris-versicolour'
    else:
        label = 'Iris-virginica'
    return label
