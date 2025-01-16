function algorithm(ẑₙ,OoS)
    list_G = Float64[]
    list_v = Float64[]
    for k in 1:30
        print(".")
        ending = k*1000        
        start = ending - 999
        #@show start
        #@show ending

        data_Dᵏ = OoS[:,start:ending]

        #v̄ᵏ
        v̄ᵏ =  fullInfoSAA(data_Dᵏ,1000,CostMatrix) #fullInfoSAA(1000,J,I,data_Dᵏ,cz,qw,ρ,μᵢⱼ)

        #v̂ᵏ
        v̂ᵏ = solution_cost(ẑₙ,data_Dᵏ,1000,CostMatrix) #solution_cost(ẑₙ,1000,J,I,data_Dᵏ,cz,qw,ρ,μᵢⱼ)

        #Ĝᵏ = v̂ᵏ - v̄ᵏ
        Ĝᵏ = v̂ᵏ - v̄ᵏ
        #@show Ĝᵏ
        push!(list_G,Ĝᵏ)
        push!(list_v,v̄ᵏ)
    end

    #Construct normalized estimate of the 99% UCB on the optimal_gap
    v̄ = mean(list_v)
    B̂ = (100/abs(v̄))*(  (1/30)*sum( list_G[k] + 2.462*sqrt( (var(list_G)/30 ))  for k in 1:30) )
    println("")
    return B̂
end

function solution_cost(Z,Yjk,K,CostMatrix)
    dz = 4 #warehouses 
    dy = 12 #Client locations
    p₁ = 5  #Costo de producir antes
    p₂ = 100 #costo de producción en el ultimo minuto

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variables(model, begin
        t[1:dz,1:K] #recourse variable ti (lo que produzco en el ultimo min en el warehouse i)
        S[1:dz,1:dy,1:K] #Cantidad transportada del warehouse i al cliente j
    end
    )

    @objective(model,Min,
            #         (1/K)*Σₖ( p1*Σᵢ{Zᵢ}         +         p2*Σᵢ{tᵢₖ}                 +             #Cij*Σᵢⱼ{Sᵢⱼ})
            (1/K).*sum( p₁*sum(Z[i] for i in 1:dz) + ( p₂*sum(t[i,k] for i in 1:dz) +  sum(sum( CostMatrix[i,j]*S[i,j,k] for j in 1:dy) for i in 1:dz) ) for k in 1:K)   
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
    #print("Optimization termination status: ")
    #printstyled(termination_status(model), "\n"; bold = true, color = :blue)

    return objective_value(model)
end


function fullInfoSAA(Yjk,K,CostMatrix)
    dz = 4 #warehouses 
    dy = 12 #Client locations
    p₁ = 5  #Costo de producir antes
    p₂ = 100 #costo de producción en el ultimo minuto

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
            (1/K).*sum( p₁*sum(Z[i] for i in 1:dz) + ( p₂*sum(t[i,k] for i in 1:dz) +  sum(sum( CostMatrix[i,j]*S[i,j,k] for j in 1:dy) for i in 1:dz) ) for k in 1:K)   
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
    #print("Optimization termination status: ")
    #printstyled(termination_status(model), "\n"; bold = true, color = :blue)

    return objective_value(model)
end
