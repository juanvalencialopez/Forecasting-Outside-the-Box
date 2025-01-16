function LS(Y,X,J,L)
    #β = (XᵗX)^(-1)(Xᵗy)
    θₗₛ = ones(J,L+1)*10000
    for j in 1:J
        yⱼ = Y[j,:]
        β = coef(linregress(X,yⱼ)) #py"LR"(X,yⱼ)
        #β = append!(coef,inter)
        θₗₛ[j,:] = β
    end
    return θₗₛ
end
