residuos(forecast,data_real)  = data_real .- forecast



function forecast_cesgado(θⱼ,X,L,J,T)
    #Ŷ = ones(T)

    ŷⱼ = sum(θⱼ[l].*X[:,l] for l in 1:L) .+ θⱼ[L+1]

    #=
    for j in 1:J
        #Obtengo thetas del cliente j
        θⱼ = θ[j,:]
        #Obtengo las predicciones para todos los samples i
        
        Ŷ[:,j] = ŷⱼ
    end
    =#
    return ŷⱼ
end
