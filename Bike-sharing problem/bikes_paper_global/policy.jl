function net_multiple_demands(T,I_cli,stations_id,Y)
    y = ones(T,I_cli-1)*999 
    withd_y = ones(T,I_cli-1)*999 
    return_y = ones(T,I_cli-1)*999 
    for t in 1:T
        y_i = Y[t]
        y_i_netas,y_i_return, y_i_withd = py"""net_demand"""(y_i,stations_id)
        y[t,:] =   y_i_netas
        return_y[t,:] =   y_i_return  
        withd_y[t,:] =   y_i_withd    
    end
    @show size(y)
    @show size(withd_y)
    @show size(return_y)
    return y,return_y,withd_y
end







function bikes_sharing(I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,pᵢ,cᵢ,ratio_gᵢ,ratio_hᵢ,gᵢ,hᵢ)
    model_1s = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_1s, "OutputFlag", 0)
    @variables(model_1s,
    begin
        xᵢ[1:(I-1)] >=0
        aᵢ[1:(I-1)] >=0
        bᵢ[1:(I-1)] >=0
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
    
    
    @objective(model_1s,Min,
    sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1)) + sum(cᵢⱼ[i]*vᵢⱼ[i,i+1] - pᵢ[i]*zᵢ[i] +pos_Eᵢ[i]*cᵢ[i]  + pos_Bᵢ[i]*(cᵢ[i]/Qᵢ[i])  
    + pᵢ[i]*aᵢ[i] + cᵢ[i]*bᵢ[i] for i in 1:(I-1)) #+ 10*yᵢ[i]
    )

    
    @constraints(model_1s,
    begin

        #New constraints
        [i in 1:(I-1)], xᵢ[i] ≥ gᵢ[i]*ratio_gᵢ[i] - aᵢ[i]

        [i in 1:(I-1)], Qᵢ[i] - xᵢ[i] ≥ hᵢ[i]*ratio_hᵢ[i] - bᵢ[i]

        # 1 
        #   Impose that the delivered quantity to each station has to be at least as great as the initial requirement at that station
        #[i in 1:(I-1)], xᵢ[i] ≥ x̄ᵢ[i]

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
    optimize!(model_1s)
    #@show value.(pos_Eᵢ)
    #@show value.(Eᵢ)
    #@show value.(pos_Bᵢ)
    #@show value.(Bᵢ)
    #@show value.(vᵢⱼ)
    #@show value.(xᵢ)
    #@show ξ
    #@show value.(aᵢ)
    #@show value.(bᵢ)
    return value.(xᵢ),objective_value(model_1s)
end


function bikes_sharing2(xᵢ,I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,pᵢ,cᵢ,ratio_gᵢ,ratio_hᵢ,gᵢ,hᵢ)
    model_test = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_test, "OutputFlag", 0)

    @variables(model_test,
    begin
        #xᵢ[1:(I-1)] >=0
        aᵢ[1:(I-1)] >=0
        bᵢ[1:(I-1)] >=0
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

    @objective(model_test,Min,
    sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1)) + sum(cᵢⱼ[i]*vᵢⱼ[i,i+1] - pᵢ[i]*zᵢ[i] +pos_Eᵢ[i]*cᵢ[i]  + pos_Bᵢ[i]*(cᵢ[i]/Qᵢ[i]) 
    + pᵢ[i]*aᵢ[i] + cᵢ[i]*bᵢ[i] for i in 1:(I-1)) #+10*yᵢ[i]
    )

    
    @constraints(model_test,
    begin
        #New constraints
        [i in 1:(I-1)], xᵢ[i] ≥ gᵢ[i]*ratio_gᵢ[i] - aᵢ[i]

        [i in 1:(I-1)], Qᵢ[i] - xᵢ[i] ≥ hᵢ[i]*ratio_hᵢ[i] - bᵢ[i]

        # 1 
        #   Impose that the delivered quantity to each station has to be at least as great as the initial requirement at that station
        #[i in 1:(I-1)], xᵢ[i] ≥ x̄ᵢ[i]

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

    optimize!(model_test)
    #@show value.(vᵢⱼ)
    return objective_value(model_test)

end


function bikes_sharing_SAA(T,I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,pᵢ,cᵢ, ratio_gᵢ,ratio_hᵢ,gᵢ,hᵢ)
    model_saa = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_saa, "OutputFlag", 0)
    @variables(model_saa,
    begin
        xᵢ[1:(I-1)] >=0
        aᵢ[1:(I-1),1:T] >=0
        bᵢ[1:(I-1),1:T] >=0
        vᵢⱼ[i = 1:(I-1),i+1,1:T]   >=0
        Iᵢ[1:(I-1),1:T]
        Iₙ[[35],1:T]  
        zᵢ[1:(I-1),1:T] <=0
        yᵢ[1:(I-1),1:T] >=0
        Eᵢ[1:(I-1),1:T]
        pos_Eᵢ[1:(I-1),1:T] >=0
        Bᵢ[1:(I-1),1:T]
        pos_Bᵢ[1:(I-1),1:T]>=0
    end
    )
    
    
    @objective(model_saa,Min,
    (1/T)*sum( sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1)) + sum(cᵢⱼ[i]*vᵢⱼ[i,i+1,t] - pᵢ[i]*zᵢ[i,t] +pos_Eᵢ[i,t]*cᵢ[i]  + pos_Bᵢ[i,t]*(cᵢ[i]/Qᵢ[i]) 
    + pᵢ[i]*aᵢ[i,t] + cᵢ[i]*bᵢ[i,t] for i in 1:(I-1) ) for t in 1:T))

    
    @constraints(model_saa,
    begin

        #New constraints
        [i in 1:(I-1),t in 1:T], xᵢ[i] ≥ gᵢ[t,i]*ratio_gᵢ[i] - aᵢ[i,t]

        [i in 1:(I-1),t in 1:T], Qᵢ[i] - xᵢ[i] ≥ hᵢ[t,i]*ratio_hᵢ[i] - bᵢ[i,t]


        # 1 
        #   Impose that the delivered quantity to each station has to be at least as great as the initial requirement at that station
        #[i in 1:(I-1)], xᵢ[i] ≥ x̄ᵢ[i]

        #2  
        #   Guarantee that the sum between th initially available  and the  quantity allocated
        #   at each station does not exceed the station capacity
        [i in 1:(I-1)], Īᵢₒ[i] + xᵢ[i] ≤ Qᵢ[i]

        # 3
        #   implies that the total number of delivered bikes to stations is less than the available quantity at the depot
        sum(xᵢ[i] for i in 1:(I-1)) ≤ Īₙ₀

        # 4
        #   ensure that the number of bikes carried by the vehicle during rebalancing never exceeds its capacity
        [i in 1:(I-1),t in 1:T], vᵢⱼ[i,i+1,t] ≤ C

        # 5 
        #   ensure that, for the depot,
        #   the quantity at the end of the period is equal to the initial bike availability and 
        #   the quantity received from the last visited station minus the quantities delivered to stations
        
        [t in 1:T], Iₙ[I,t] == Īₙ₀ - sum(xᵢ[i] for i in 1:(I-1)) + vᵢⱼ[I-1,I,t]

        #6 
        #   ensure that, at the end of the rebalancing period, the number of
        #   bikes at the depot does not exceed its capacity
        [t in 1:T], Iₙ[35,t] ≤ Īₙ₀
        
        #7
        # ensure that, for the first visited station, 
        #the quantity at the end of rebalancing is equal to the sum between the initial available quantity 
        #and the quantity received from the depot minus the quantities used to satisfy the demand 
        #and those bikes that are redistributed to subsequent stations on the route
       
        [t in 1:T], Iᵢ[1,t] == Īᵢₒ[1] + xᵢ[1] - ξ[t,1] - vᵢⱼ[1,2,t]
        [t in 1:T], -zᵢ[1,t] + Iᵢ[1,t] >=0
        [t in 1:T], yᵢ[1,t] - Iᵢ[1,t] >=0
         
        #8 
        #   determine the inventory position (which can be negative or positive) at a station other than the first, 
        #   as a function of the initial inventory level, the number allocated, the number withdrawn/returned, 
        #   and the number redistributed to another station in each scenario s ∈ S
        [i in 2:(I-1),t in 1:T], Iᵢ[i,t] == Īᵢₒ[i] + xᵢ[i] - ξ[t,i] + vᵢⱼ[i-1,i,t] - vᵢⱼ[i,i+1,t] 
        [i in 2:(I-1),t in 1:T], -zᵢ[i,t] + Iᵢ[i,t] >=0
        [i in 2:(I-1),t in 1:T], yᵢ[i,t] - Iᵢ[i,t] >=0
        
        #---
        [i in 1:(I-1),t in 1:T], Eᵢ[i,t] == yᵢ[i,t] - Qᵢ[i]
        [i in 1:(I-1),t in 1:T], pos_Eᵢ[i,t] - Eᵢ[i,t] >=0

        #---
        [i in 1:(I-1),t in 1:T], Bᵢ[i,t] == yᵢ[i,t] - xᵢ[i] - Īᵢₒ[i] -  pos_Eᵢ[i,t]
        [i in 1:(I-1),t in 1:T], pos_Bᵢ[i,t] - Bᵢ[i,t] >=0
        
    end)   
    #print(model_saa) 
    optimize!(model_saa)
    return value.(xᵢ),objective_value(model_saa)

end
