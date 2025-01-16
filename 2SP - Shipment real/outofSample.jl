
function out_of_sample(zₜ,yₜ)
    dz = 4
    dy = 12
    p₁ = 5  #Costo de producir antes
    p₂ = 100 #costo de producción en el ultimo minuto
    DistanceMatrix = transpose([0.15  1.3124  1.85  1.3124;0.50026  0.93408  1.7874  1.6039; 0.93408  0.50026  1.6039  1.7874;1.3124  0.15  1.3124  1.85; 1.6039  0.50026  0.93408  1.7874;1.7874  0.93408  0.50026  1.6039; 1.85  1.3124  0.15  1.3124;1.7874  1.6039  0.50026  0.93408; 1.6039  1.7874  0.93408  0.50026; 1.3124  1.85  1.3124  0.15;0.93408  1.7874  1.6039  0.50026;0.50026  1.6039  1.7874  0.93408])
    CostMatrix = 10*DistanceMatrix
    CostMatrix = CostMatrix[:,1:dy]
    

    model = Model(Gurobi.Optimizer)

    @variables(model,
    begin
        tᵢ[1:dz] >=0
        sᵢⱼ[1:dz,1:dy] >=0
    end)

    @objective(model,Min,
        p₁*sum(zₜ) + ( p₂*sum(tᵢ[i] for i in 1:dz) + sum(CostMatrix[i,j]*sᵢⱼ[i,j] for i in 1:dz,j in 1:dy )  )
    )


    @constraints(model,
    begin
        [j in 1:dy], sum(sᵢⱼ[i,j] for i in 1:dz) >= yₜ[j]
        [i in 1:dz], sum(sᵢⱼ[i,j] for j in 1:dy ) <= zₜ[i] + tᵢ[i]     
    end)

    JuMP.optimize!(model)
    #println(model)
    return objective_value(model)
    
end






#=
function expectedCost(z1,z2,z3,z4,z5,z6,z7,samples)
    x,y = size(samples)
    list_costs_1 = Float64[]
    list_costs_2 = Float64[]
    list_costs_3 = Float64[]
    list_costs_4 = Float64[]
    list_costs_5 = Float64[]
    list_costs_6 = Float64[]
    list_costs_7 = Float64[]


    for i in 1:y
        Demand_OoS_i = samples[:,i]
        TrueCost_1 = out_of_sample(z1,Demand_OoS_i)
        TrueCost_2 = out_of_sample(z2,Demand_OoS_i)
        TrueCost_3 = out_of_sample(z3,Demand_OoS_i)
        TrueCost_4 = out_of_sample(z4,Demand_OoS_i)
        TrueCost_5 = out_of_sample(z5,Demand_OoS_i)
        TrueCost_6 = out_of_sample(z6,Demand_OoS_i)
        TrueCost_7 = out_of_sample(z7,Demand_OoS_i)

        append!(list_costs_1,TrueCost_1)
        append!(list_costs_2,TrueCost_2)
        append!(list_costs_3,TrueCost_3)
        append!(list_costs_4,TrueCost_4)
        append!(list_costs_5,TrueCost_5)
        append!(list_costs_6,TrueCost_6)
        append!(list_costs_7,TrueCost_7)
    end

    return mean(list_costs_1),mean(list_costs_2),mean(list_costs_3),mean(list_costs_4),mean(list_costs_5),mean(list_costs_6),mean(list_costs_7)
end
=#