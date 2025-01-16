function bertProblem(predicted_demand)
    dy = 12
    dz = 4
    p₁ = 5  #Costo de producir antes
    p₂ = 100 #costo de producción en el ultimo minuto
    DistanceMatrix = transpose([0.15  1.3124  1.85  1.3124;0.50026  0.93408  1.7874  1.6039; 0.93408  0.50026  1.6039  1.7874;1.3124  0.15  1.3124  1.85; 1.6039  0.50026  0.93408  1.7874;1.7874  0.93408  0.50026  1.6039; 1.85  1.3124  0.15  1.3124;1.7874  1.6039  0.50026  0.93408; 1.6039  1.7874  0.93408  0.50026; 1.3124  1.85  1.3124  0.15;0.93408  1.7874  1.6039  0.50026;0.50026  1.6039  1.7874  0.93408])
    CostMatrix = 10*DistanceMatrix
    CostMatrix = CostMatrix[:,1:dy]
    

    model_1 = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_1, "OutputFlag", 0)

    @variables(model_1,
    begin
        zᵢ[1:dz] >=0
        tᵢ[1:dz] >=0
        sᵢⱼ[1:dz,1:dy] >=0
    end)

    @objective(model_1,Min,
        p₁*sum(zᵢ[i] for i in 1:dz) + ( p₂*sum(tᵢ[i] for i in 1:dz) + sum(CostMatrix[i,j]*sᵢⱼ[i,j] for i in 1:dz,j in 1:dy )  )
    )


    @constraints(model_1,
    begin
        [j in 1:dy], sum(sᵢⱼ[i,j] for i in 1:dz) >= predicted_demand[j]
        [i in 1:dz], sum(sᵢⱼ[i,j] for j in 1:dy ) <= zᵢ[i] + tᵢ[i]     
    end)

    JuMP.optimize!(model_1)
    return value.(zᵢ)
end
