function LS(Y,X,J,L)
    #β = (XᵗX)^(-1)(Xᵗy)
    θₗₛ = coef(linregress(X,Y))
    return θₗₛ
end


