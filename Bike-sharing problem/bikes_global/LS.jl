function LS(Y,X,J,L)
    #β = (XᵗX)^(-1)(Xᵗy)
    θₗₛ = coef(linregress(X,Y))
    return θₗₛ
end


#=
py"""
import numpy as np
from sklearn.linear_model import LinearRegression

def LR(X,y):
    reg = LinearRegression().fit(X, y)
    return reg, reg.coef_ ,reg.intercept_



def predict_LR(regressor,x_new):
    x_new   = np.array(x_new).reshape(1, -1)
    y_pred = regressor.predict(x_new)
    return y_pred
"""



function LS(Y,X,J,L)
    θₗₛ = ones(J,L+1)*10000
    for j in 1:J
        yⱼ = Y[:,j]
        coef,inter = py"LR"(X,yⱼ)
        β = append!(coef,inter)
        θₗₛ[j,:] = β
    end
    return θₗₛ
end



function pointPred(X,θ,J,L)
    
    ŷ_pred = zeros(J)
    for j in 1:J
        ŷ_pred[j] = sum(θ[j,l].*X[l] for l in 1:(L-1) ) .+ θ[j,L]
    end
    return ŷ_pred
end
=#