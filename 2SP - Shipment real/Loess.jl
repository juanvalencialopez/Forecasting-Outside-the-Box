
function I_f(hₙ,d)
    if d<=hₙ
        return 1
    else
        return 0
    end
end

function k_funct(X,X_fresh,h)
    di = norm(X .- X_fresh,2)
    kᵢ = ( (1 - (di/h)^3 )^3 )*I_f(h,di)
    return kᵢ
end

function Sigma(X,ns,X_fresh,h)
    #sigmaSum = Float64[]
    sigmaSum = [ 0.0 0.0 0.0; 0.0 0.0 0.0;0.0 0.0 0.0]
    for i in 1:ns
        Xi = X[:,i]
        kᵢ = k_funct(Xi,X_fresh,h)
        dist = Xi .- X_fresh
        sig = kᵢ*dist*dist'
        sigmaSum = sigmaSum + sig

    end
    return sigmaSum
end

function LOESS(Xtrain,ns,X_fresh)

    hₙ = Int(round(10*(ns^-0.25)))
    

    
    #print(Xtrain)
    betas = Float64[]
    for i in 1:ns
        println(i)
        Xi = Xtrain[:,i]
        
        kᵢ = k_funct(Xi,X_fresh,hₙ) 

        s = 0
        for j in 1:ns
            
            Xj = Xtrain[:,j]
            kⱼ = k_funct(Xj,X_fresh,hₙ)

            izq = (Xtrain[:,j] .- X_fresh)'
            der =  Xtrain[:,i] .- X_fresh   
            arg = kⱼ*(izq*inv(Sigma(Xtrain,ns,X_fresh,hₙ))*der)
            s = s + arg
            #println(s)
        end

        derecha = maximum((1-s,0))
        finalArg = kᵢ*derecha
        #println(finalArg)
        append!(betas,finalArg)
        
    end
    return betas
    
end







function LOESSopt(Yjk,K,βs,CostMatrix)
    dz = 4 #warehouses 
    dy = 3 #Client locations
    p₁ = 5  #Costo de producir antes
    p₂ = 100 #costo de producción en el ultimo minuto


    #Cantidad de escenarios

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variables(model, begin
        Z[1:dz] #cantidad a producir en cada warehouse i
        t[1:dz,1:K] #recourse variable ti (lo que produzco en el ultimo min en el warehouse i)
        S[1:dz,1:dy,1:K] #Cantidad transportada del warehouse i al cliente j
    end
    )

    @objective(model,Min,
            #         (1/K)*Σₖ( p1*Σᵢ{Zᵢ}         +         p2*Σᵢ{tᵢₖ}                 +             #Cij*Σᵢⱼ{Sᵢⱼ})
            (1/K).*sum( βs[k].*(p₁*sum(Z[i] for i in 1:dz) + ( p₂*sum(t[i,k] for i in 1:dz) +  sum(sum( CostMatrix[i,j]*S[i,j,k] for j in 1:dy) for i in 1:dz) )) for k in 1:K)   
    )


    #Lo que le mando desde i a j, es >= a la demanda de j. Para cada escenario k
    @constraint(model,
        [j in 1:dy,k in 1:K], sum(S[i,j,k] for i in 1:dz) >= Yjk[j,k]
    )


    #Lo que le mando desde i a j es menor que lo que tenia producido + lo que produzco en el ultimo min
    @constraint(model,
        [i in 1:dz,k in 1:K], sum(S[i,j,k] for j in 1:dy) <= Z[i] + t[i,k]
    )


    #Producción no neg
    @constraint(model,
        [i in 1:dz], Z[i] >=0
    )

    #Producción ultimo min no neg 
    @constraint(model,
        [i in 1:dz,k in 1:K], t[i,k] >=0
    )

    #cantidad transportada no neg
    @constraint(model,
        [i in 1:dz,j in 1:dy, k in 1:K], S[i,j,k] >=0
    )


    JuMP.optimize!(model)
    print("Optimization termination status: ")
    printstyled(termination_status(model), "\n"; bold = true, color = :blue)

    return value.(Z)
end
