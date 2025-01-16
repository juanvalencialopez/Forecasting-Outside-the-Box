py"""
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def getLeav(regressor,Xnew,X,y):

    Xnew = Xnew.reshape(1, -1)
    #print("prediccion: ",regressor.predict(Xnew))
    idx = regressor.apply(Xnew)[0]
    #print("Nodo en el que term  ina el nuevo feature (idx)",idx)

    children_left = regressor.tree_.children_left
    children_right = regressor.tree_.children_right
    #Basado en https://datascience.stackexchange.com/questions/87122/how-to-obtain-the-final-values-of-a-decisiontreeregressor-in-scikit-learn
    leaf_nodes = []
    for i in range(len(children_left)):
        if children_left[i] == children_right[i]:
            leaf_nodes.append(i)

    all_val = regressor.tree_.value[leaf_nodes,0,0] 

    on_leaf = regressor.apply(X)
    x_leaves = []
    y_leaves = []
    index = leaf_nodes.index(idx)
    
    for i in np.unique(on_leaf):
        x_leaves.append(X[np.argwhere(on_leaf==i)]) 
        y_leaves.append(y[np.argwhere(on_leaf==i)]) 

    return x_leaves[index], y_leaves[index],regressor.predict(Xnew)


def getRegressor(X_train,y_train,X_test,y_test):
    regressor = DecisionTreeRegressor(max_depth = 8, min_samples_leaf = 1, random_state = 2).fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    print("score ", regressor.score(X_test, y_test))
    print("square ",mean_squared_error(y_test, y_pred))
    print(cross_val_score(regressor, X_test, y_test, cv=5,scoring="neg_mean_squared_error" ))
    return regressor


"""
