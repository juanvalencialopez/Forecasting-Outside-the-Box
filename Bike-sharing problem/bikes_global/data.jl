
py"""
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from pandas.core.tools.datetimes import to_datetime

def bringData(X_path,y_path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train.reset_index(drop=True).to_numpy(), X_test.to_numpy(), y_train.reset_index(drop=True).to_numpy(), y_test.to_numpy()

    
def sample_train_data(Nsample,X_train,y_train):

    idx = np.random.choice(np.arange(len(X_train)), Nsample, replace=False)
    sample_Xtrain = X_train.iloc[idx,:]
    sample_ytrain = y_train.iloc[idx,:]

    #X_sample = X_train.sample(n=Nsample)

    return sample_Xtrain.to_numpy() ,sample_ytrain.to_numpy()
"""

#X_train, X_test, X_val, y_train, y_test, y_val = py"bringData"("X_pos_demands.csv","y_pos_demands.csv")
#reg = py"getRegressor"(X_train,y_train,X_test,y_test)
#=
X_train, X_test, X_val, y_train, y_test, y_val = py"bringData"("SeoulBikeData_xlsx.xlsx")

reg = py"getRegressor"(X_train,y_train,X_test,y_test)
py"getLeav"(reg,X_val[1,:],X_train,y_train)

getLeav = py"getLeav"(reg,X_val[1,:],X_train,y_train)
X_hoja = reshape(getLeav[1][:,:,:],(length(getLeav[1][:,:,1]),13))
Y_hoja = reshape(getLeav[2][:,:,:],(length(getLeav[2][:,:,1]),10))
=#

