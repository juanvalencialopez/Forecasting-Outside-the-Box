

function bike_PlanPolicy(I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,pᵢ,cᵢ)
    modelK1 = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(modelK1, "OutputFlag", 0)

    @variables(modelK1,
    begin
        xᵢ[1:(I-1)] >=0
        vᵢⱼ[i = 1:(I-1),i+1]   >=0
        Iᵢ[1:(I-1)]
        Iₙ[[35]]  
        zᵢ[1:(I-1)] <=0
        yᵢ[1:(I-1)] >=0
        Eᵢ[1:(I-1)]
        pos_Eᵢ[1:(I-1)] >=0
        Bᵢ[1:(I-1)]
        pos_Bᵢ[1:(I-1)]>=0
    end
    )
    
    
    @objective(modelK1,Min,
    sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1))+ sum(cᵢⱼ[i]*vᵢⱼ[i,i+1] - pᵢ[i]*zᵢ[i] +pos_Eᵢ[i]*cᵢ[i]  + pos_Bᵢ[i]*(cᵢ[i]/Qᵢ[i])  for i in 1:(I-1))
    )

    
    @constraints(modelK1,
    begin
        # 1 
        #   Impose that the delivered quantity to each station has to be at least as great as the initial requirement at that station
        [i in 1:(I-1)], xᵢ[i] ≥ x̄ᵢ[i]

        #2  
        #   Guarantee that the sum between th initially available  and the  quantity allocated
        #   at each station does not exceed the station capacity
        [i in 1:(I-1)], Īᵢₒ[i] + xᵢ[i] ≤ Qᵢ[i]

        # 3
        #   implies that the total number of delivered bikes to stations is less than the available quantity at the depot
        sum(xᵢ[i] for i in 1:(I-1)) ≤ Īₙ₀

        # 4
        #   ensure that the number of bikes carried by the vehicle during rebalancing never exceeds its capacity
        [i in 1:(I-1)], vᵢⱼ[i,i+1] ≤ C

        # 5 
        #   ensure that, for the depot,
        #   the quantity at the end of the period is equal to the initial bike availability and 
        #   the quantity received from the last visited station minus the quantities delivered to stations
        Iₙ[35] == Īₙ₀ - sum(xᵢ[i] for i in 1:(I-1)) + vᵢⱼ[34,35]

        #6 
        #   ensure that, at the end of the rebalancing period, the number of
        #   bikes at the depot does not exceed its capacity
        Iₙ[35] ≤ Īₙ₀

        #7
        # ensure that, for the first visited station, 
        #the quantity at the end of rebalancing is equal to the sum between the initial available quantity 
        #and the quantity received from the depot minus the quantities used to satisfy the demand 
        #and those bikes that are redistributed to subsequent stations on the route
        Iᵢ[1] == Īᵢₒ[1] + xᵢ[1] - ξ[1] - vᵢⱼ[1,2]
        - zᵢ[1] + Iᵢ[1] >=0
        yᵢ[1] - Iᵢ[1] >=0

        #8 
        #   determine the inventory position (which can be negative or positive) at a station other than the first, 
        #   as a function of the initial inventory level, the number allocated, the number withdrawn/returned, 
        #   and the number redistributed to another station in each scenario s ∈ S
        [i in 2:(I-1)], Iᵢ[i] == Īᵢₒ[i] + xᵢ[i] - ξ[i] + vᵢⱼ[i-1,i] - vᵢⱼ[i,i+1] 
        [i in 2:(I-1)], - zᵢ[i] + Iᵢ[i] >=0
        [i in 2:(I-1)], yᵢ[i] - Iᵢ[i] >=0

        #---
        [i in 1:(I-1)], Eᵢ[i] == yᵢ[i] - Qᵢ[i]
        [i in 1:(I-1)], pos_Eᵢ[i] - Eᵢ[i] >=0

        #---
        [i in 1:(I-1)], Bᵢ[i] == yᵢ[i] - xᵢ[i] - Īᵢₒ[i] -  pos_Eᵢ[i]
        [i in 1:(I-1)], pos_Bᵢ[i] - Bᵢ[i] >=0

    end)    
    optimize!(modelK1)
    return value.(xᵢ)

end


function bike_CostAssessment(xᵢ,I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,pᵢ,cᵢ)
    modelK2 = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(modelK2, "OutputFlag", 0)

    @variables(modelK2,
    begin
        #xᵢ[1:(I-1)] >=0
        vᵢⱼ[i = 1:(I-1),i+1]   >=0
        Iᵢ[1:(I-1)]
        Iₙ[[35]] 
        zᵢ[1:(I-1)] <=0
        yᵢ[1:(I-1)] >=0
        Eᵢ[1:(I-1)]
        pos_Eᵢ[1:(I-1)] >=0
        Bᵢ[1:(I-1)]
        pos_Bᵢ[1:(I-1)]>=0
    end
    )

    @objective(modelK2,Min,
    sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1)) + sum(cᵢⱼ[i]*vᵢⱼ[i,i+1] - pᵢ[i]*zᵢ[i] +pos_Eᵢ[i]*cᵢ[i]  + pos_Bᵢ[i]*(cᵢ[i]/Qᵢ[i]) for i in 1:(I-1)) 
    )

    
    @constraints(modelK2,
    begin
        # 1 
        #   Impose that the delivered quantity to each station has to be at least as great as the initial requirement at that station
        [i in 1:(I-1)], xᵢ[i] ≥ x̄ᵢ[i]

        #2  
        #   Guarantee that the sum between th initially available  and the  quantity allocated
        #   at each station does not exceed the station capacity
        [i in 1:(I-1)], Īᵢₒ[i] + xᵢ[i] ≤ Qᵢ[i]

        # 3
        #   implies that the total number of delivered bikes to stations is less than the available quantity at the depot
        sum(xᵢ[i] for i in 1:(I-1)) ≤ Īₙ₀

        # 4
        #   ensure that the number of bikes carried by the vehicle during rebalancing never exceeds its capacity
        [i in 1:(I-1)], vᵢⱼ[i,i+1] ≤ C

        # 5 
        #   ensure that, for the depot,
        #   the quantity at the end of the period is equal to the initial bike availability and 
        #   the quantity received from the last visited station minus the quantities delivered to stations
        Iₙ[I] == Īₙ₀ - sum(xᵢ[i] for i in 1:(I-1)) + vᵢⱼ[I-1,I]

        #6 
        #   ensure that, at the end of the rebalancing period, the number of
        #   bikes at the depot does not exceed its capacity
        Iₙ[I] ≤ Īₙ₀

        #7
        # ensure that, for the first visited station, 
        #the quantity at the end of rebalancing is equal to the sum between the initial available quantity 
        #and the quantity received from the depot minus the quantities used to satisfy the demand 
        #and those bikes that are redistributed to subsequent stations on the route
        Iᵢ[1] == Īᵢₒ[1] + xᵢ[1] - ξ[1] - vᵢⱼ[1,2]
        - zᵢ[1] + Iᵢ[1] >=0
        yᵢ[1] - Iᵢ[1] >=0
        #8 
        #   determine the inventory position (which can be negative or positive) at a station other than the first, 
        #   as a function of the initial inventory level, the number allocated, the number withdrawn/returned, 
        #   and the number redistributed to another station in each scenario s ∈ S
        [i in 2:(I-1)], Iᵢ[i] == Īᵢₒ[i] + xᵢ[i] - ξ[i] + vᵢⱼ[i-1,i] - vᵢⱼ[i,i+1] 
        [i in 2:(I-1)], - zᵢ[i] + Iᵢ[i] >=0
        [i in 2:(I-1)], yᵢ[i] - Iᵢ[i] >=0

        #---
        [i in 1:(I-1)], Eᵢ[i] == yᵢ[i] - Qᵢ[i]
        [i in 1:(I-1)], pos_Eᵢ[i] - Eᵢ[i] >=0

        #---
        [i in 1:(I-1)], Bᵢ[i] == yᵢ[i] - xᵢ[i] - Īᵢₒ[i] -  pos_Eᵢ[i]
        [i in 1:(I-1)], pos_Bᵢ[i] - Bᵢ[i] >=0

    end)

    optimize!(modelK2)
    return objective_value(modelK2)

end

function forecast(θ,Xₜ,L)

    #x,y = size(θ) #x-> filas , y = 2 por t1 y t2
    ŷⱼₜ = sum(θ[l]*Xₜ[l] for l in 1:L) + θ[L+1]
    #=
    d_list = Float64[]
    for j in 1:x
       if  ŷⱼₜ[j]<0
        push!(d_list,abs(ŷⱼₜ[j]))
        ŷⱼₜ[j] = 0
       else
        push!(d_list,0)
       end 
    end
    #@show ŷⱼₜ
    #@show d_list
    =#
    return ŷⱼₜ #,d_list
end

function heuristicAD_par(θ,T,data_xₜ,data_yₜ,I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,pᵢ,cᵢ,L,station_ids)

    #println("solving NM paralelo....")
    #print(".")
    cost_θ = Float64[] #[Float64[] for _ in 1:Threads.nthreads()] #Float64[]
    
    for t in 1:T #Threads.@threads for t in 1:T
        #----- forecast, ŷₜ = ψ(Θ,xₜ)
        #println("forecasting....")
  
        ŷⱼₜ = forecast(θ,data_xₜ[t,:],L) 
        ŷⱼₜ = py"""net_demand"""(ŷⱼₜ,station_ids)
        #@show ŷⱼₜ
        
        #------ Plan Policy, zₜ* ∈ arg min Gₚ(z,ŷₜ)

        zₜ = bike_PlanPolicy(I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ŷⱼₜ,pᵢ,cᵢ)
        #zₜ = kannanPlanPolicy(J,I,ŷⱼₜ ,cz,qw,ρ,μᵢⱼ)
        
        
        #------ Cost assesmet costₜ ∈ Gₐ(z,yₜ)
        #print(".")
        #println("Cost assessment...")
        data_y_neta = py"""net_demand"""(data_yₜ[t],station_ids) 
        costₜ = bike_CostAssessment(zₜ,I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,data_y_neta,pᵢ,cᵢ)
        #costₜ = kannanCostAssessment(zₜ,J,I,data_yⱼₜ[:,t],cz,qw,ρ,μᵢⱼ,δ)
        
        #@show costₜ
        #push!(cost_θ[Threads.threadid()], costₜ)
        push!(cost_θ, costₜ)
        #append!(cost_θ,costₜ)
        
    end

    costTotal = sum(cost_θ)/T
    #@show costTotal
    return costTotal
    
end
