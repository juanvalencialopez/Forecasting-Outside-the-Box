function net_multiple_demands(T,I_cli,stations_id,Y)
    y = ones(T,I_cli-1)*999 

    for t in 1:T
        y_i = Y[t]
        y_i_netas = py"""net_demand"""(y_i,stations_id)
        y[t,:] =   y_i_netas  
    end
    return y
end







function bikes_sharing(I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,pᵢ,cᵢ)
    model_1s = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_1s, "OutputFlag", 0)
    @variables(model_1s,
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
    
    
    @objective(model_1s,Min,
    sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1)) + sum(cᵢⱼ[i]*vᵢⱼ[i,i+1] - pᵢ[i]*zᵢ[i] +pos_Eᵢ[i]*cᵢ[i]  + pos_Bᵢ[i]*(cᵢ[i]/Qᵢ[i])  for i in 1:(I-1)) #+ 10*yᵢ[i]
    )

    
    @constraints(model_1s,
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
    optimize!(model_1s)
    @show value.(xᵢ)
    @show ξ
    return value.(xᵢ),objective_value(model_1s)
end


function bikes_sharing2(xᵢ,I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,pᵢ,cᵢ)
    model_test = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_test, "OutputFlag", 0)

    @variables(model_test,
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

    @objective(model_test,Min,
    sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1)) + sum(cᵢⱼ[i]*vᵢⱼ[i,i+1] - pᵢ[i]*zᵢ[i] +pos_Eᵢ[i]*cᵢ[i]  + pos_Bᵢ[i]*(cᵢ[i]/Qᵢ[i]) for i in 1:(I-1)) #+10*yᵢ[i]
    )

    
    @constraints(model_test,
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

    optimize!(model_test)
    #@show value.(vᵢⱼ)
    return objective_value(model_test)

end


function bikes_sharing_SAA(T,I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,pᵢ,cᵢ)
    model_saa = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_saa, "OutputFlag", 0)
    @variables(model_saa,
    begin
        xᵢ[1:(I-1)] >=0
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
    (1/T)*sum( sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1)) + sum(cᵢⱼ[i]*vᵢⱼ[i,i+1,t] - pᵢ[i]*zᵢ[i,t] +pos_Eᵢ[i,t]*cᵢ[i]  + pos_Bᵢ[i,t]*(cᵢ[i]/Qᵢ[i]) for i in 1:(I-1) ) for t in 1:T))

    
    @constraints(model_saa,
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

#=
function bikes_sharing_SAA_oos(xᵢ,T,I,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,pᵢ,cᵢ)
    model_saa = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model_saa, "OutputFlag", 0)
    @variables(model_saa,
    begin
        #xᵢ[1:(I-1)] >=0
        vᵢⱼ[i = 1:(I-1),i+1,1:T]   >=0
        Iᵢ[1:(I-1),1:T]
        Iₙ[[35],1:T]>=0   
        zᵢ[1:(I-1),1:T] >=0
        yᵢ[1:(I-1),1:T] >=0
        Eᵢ[1:(I-1),1:T]
        pos_Eᵢ[1:(I-1),1:T] >=0
        Bᵢ[1:(I-1),1:T]
        pos_Bᵢ[1:(I-1),1:T]>=0
    end
    )
    
    
    @objective(model_saa,Min,
    (1/T)*sum( sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1)) + sum(cᵢⱼ[i]*vᵢⱼ[i,i+1,t] + pᵢ[i]*zᵢ[i,t] +pos_Eᵢ[i,t]*cᵢ[i]  + pos_Bᵢ[i,t]*(cᵢ[i]/Qᵢ[i]) for i in 1:(I-1) ) for t in 1:T))

    
    @constraints(model_saa,
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
        [t in 1:T], zᵢ[1,t] + Iᵢ[1,t] >=0
        [t in 1:T], yᵢ[1,t] - Iᵢ[1,t] >=0
         
        #8 
        #   determine the inventory position (which can be negative or positive) at a station other than the first, 
        #   as a function of the initial inventory level, the number allocated, the number withdrawn/returned, 
        #   and the number redistributed to another station in each scenario s ∈ S
        [i in 2:(I-1),t in 1:T], Iᵢ[i,t] == Īᵢₒ[i] + xᵢ[i] - ξ[t,i] + vᵢⱼ[i-1,i,t] - vᵢⱼ[i,i+1,t] 
        [i in 2:(I-1),t in 1:T], zᵢ[i,t] + Iᵢ[i,t] >=0
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
=#

#=
I_cli = 11 # Cantidad de clientes + Depot
fᵢ = [12,9,13,11,20,13,5,7,11,10].*0.5 #Costo de alocar en un principio las bicicletas
cᵢⱼ = [12,9,13,11,20,13,5,7,11,10].*1000  #costos asociados de ir de i a j
x̄ᵢ = [0.1,0.05,0.12,0.08,0.09,0.23,0.13,0.06,0.08,0.06].*10 #Cantidad minima a alocar en estacion i en 1st stage 
Īᵢₒ = [3, 5, 2, 7, 3, 1,2, 3, 5, 1] 
Qᵢ = 120
Īₙ₀ = 1000
C = 150


N_samples = [10,100,1000,5000,10000]

#bikes_sharing_SAA(T,I_cli,fᵢ, cᵢⱼ, x̄ᵢ, Īᵢₒ, Qᵢ, Īₙ₀, C, ξ)
df_in = DataFrame()
df_out = DataFrame()
ξ_true = randn(11,10000).*10
z_true,obj_true  = bikes_sharing_SAA(10000,I_cli,fᵢ, cᵢⱼ, x̄ᵢ, Īᵢₒ, Qᵢ, Īₙ₀, C, ξ_true)
ξ = randn(11,100000).*10
for n in N_samples
    T = n
    ξ_train = ξ[:,1:T]
    z,obj_insample = bikes_sharing_SAA(T,I_cli,fᵢ, cᵢⱼ, x̄ᵢ, Īᵢₒ, Qᵢ, Īₙ₀, C, ξ_train)
    append!(df_in,DataFrame(T = string(T) , obj = obj_insample ))

    obj_oos = bikes_sharing_SAA_oos(z,T,I_cli,fᵢ, cᵢⱼ, x̄ᵢ, Īᵢₒ, Qᵢ, Īₙ₀, C, ξ_true)
    append!(df_out,DataFrame(T = string(T), obj = obj_oos, gap = abs(obj_true-obj_oos)*100/obj_true ))
end


#PlotlyJS.plot(df_out, kind="scatter", mode="lines", x=:T, y=:gap,Layout(title="%gap - Out of sample"))

=#