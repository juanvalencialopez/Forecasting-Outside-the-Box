#Función que computa los thetas iniciales para cada escenario k
function θ_iniciales(T,L,I_cli)
    θₖ⁰_dict = Dict()
    for i in 1:T
        θₖ⁰ = zeros(I_cli,L+1)
        push!(θₖ⁰_dict, i=>θₖ⁰)
    end
    return θₖ⁰_dict
end

#Función que computa los λ iniciales para cada escenario k
function λ_iniciales(T,L,I_cli)
    λₖ⁰_dict = Dict()
    for i in 1:T
        #---- Obtengo los primeros thetas
        λₖ⁰ =zeros(I_cli,L+1)
        push!(λₖ⁰_dict, i=>λₖ⁰)
    end
    return λₖ⁰_dict
end

#Función que computa el promedio para matrices en diccionario.
function theta_bar(thetas_dict,T)
    promedio_thetas  = sum(thetas_dict[i] for i in 1:T)/T
    return promedio_thetas
end

function sanity_checks(list_lang,list_thetas_pen,λₖᵛ_dict,T)
    #----Sanity checks!
    #Σλθ = 0 (en el óptimo!)   
    sanityCheck1 = mean(list_lang)  
    @show sanityCheck1
    #(alpha/2)*\xs - xbarra\^2 ->0
    sanityCheck2 = mean(list_thetas_pen)
    @show sanityCheck2 
    #Σλ = 0 
    sanityCheck3 = sum(λₖᵛ_dict[k] for k in 1:T)
    @show sanityCheck3
end

#Función que computa PH
#PARA QUE FUNCIONE: data_xₜ-> OBSERVACIONES X FEATURES | data_yₜ -> OBSERVACIONES X CLIENTES

#PH_bikes(T,L,I_cli,data_yₜ,data_xₜ,α,fᵢ,cᵢⱼ,x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,pᵢ,cᵢ)
function PH_bikes(T,L,I_cli,data_yₜ,data_xₜ,α,fᵢ,cᵢⱼ,x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,pᵢ,cᵢ)

    #------------- STEP 1. 
    #Find a xₖ⁰ ∈ argmin Fₖᶜ(xₖ), for all k ∈ K
    θₖᵛ_dict = θ_iniciales(T,L,(I_cli-1))

    #------------- STEP 2.
    #Let λₖ⁰ = 0 and x̄⁰ = Σₖpₖxₖ⁰, for all k ∈ K
    λₖᵛ_dict = λ_iniciales(T,L,(I_cli-1))
    θ_mean =  theta_bar(θₖᵛ_dict,T)

    #-------------STEP 3. 
    #set v = 0
    iterations = 1

    #------------- STEP 4. 
    #For each k ∈ K, get xₖᵛ⁺¹ by solving: xₖᵛ⁺¹ ∈ argmin Fₖᶜ(xₖ) + λₖᵛᵀxₖ + (αᵛ/2)*\xₖ - x̄ᵏ\
    dummy = 1
    δ = 10
    while dummy == 1
        #--------- 
        #Defino listas para almacenar valores (FO, Multi Lang, Thetas, FO sin penalizar)
        list_obj = Float64[]
        list_lang = Float64[]
        list_thetas_pen = Float64[]
        list_obj_sinP = Float64[]
        #---------
        #Empiezo a iterar
        for k in 1:T
            #------- Con penalización....
            #Obtengos los datos 
            yⱼₖ = data_yₜ[k,:] 
            xⱼₖ = data_xₜ[k,:]
            λₖ = λₖᵛ_dict[k]
        
            #Resuelvo el problema penalizado
            @show  yⱼₖ
            @show  xⱼₖ
            @show  θ_mean
            @show  λₖ  
            θₖᵛ⁺¹,objInSample = optimizacion_penalizada(I_cli,L,α,θ_mean,λₖ,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,yⱼₖ,xⱼₖ,pᵢ,cᵢ)

            #Updateo diccionario con thetas.
            θₖᵛ_dict[k] = θₖᵛ⁺¹ 

            #Guardo datos
            #push!(list_obj_sinP,obj_sinP)
            push!(list_thetas_pen,(α/2)*(norm(θₖᵛ⁺¹ - θ_mean))^2)
            push!(list_lang,sum(λₖ.*θₖᵛ⁺¹))
            push!(list_obj,objInSample)
        end
        #----------
        #SANITY CHECKS
        sanity_checks(list_lang,list_thetas_pen,λₖᵛ_dict,T)
        
        #---------
        #STEP 5. Compute x̄ᵛ⁺¹ = Σₖpₖxₖᵛ⁺¹
        θ_mean_2 =  theta_bar(θₖᵛ_dict,T)
        δ = (norm( θ_mean_2 .- θ_mean )^2 + (1/T)*sum( norm(θₖᵛ_dict[k] .- θ_mean_2)^2 for k in 1:T))^(0.5)
        θ_mean = θ_mean_2
        @show δ

        #---------
        #STEP 6. Check: if \xₖ - x̄ᵏ\ <epsilon for all k ∈ K: Stop!, else: get λₖᵛ⁺¹ = λₖᵛ + αᵛ(xₖᵛ⁺¹ - x̄ᵛ⁺¹)
        #vect_norms = norm.( (θₖᵛ_dict[i] - θ_mean) for i in 1:T)
        append!(df,DataFrame(i = iterations, Obj_p = mean(list_obj) ,del = Float64(δ), stop = "no", pen_1 = mean(list_lang),pen_2 = mean(list_thetas_pen)  ) )
        #append!(df,DataFrame(i = iterations, Obj_p = mean(list_obj), Obj_sp = mean(list_obj_sinP) ,del = Float64(δ), stop = "no", pen_1 = mean(list_lang),pen_2 = mean(list_thetas_pen)  ) )
        iterations = iterations+1

        if δ<= 0.1 
            println("------ En el opt")
            list_obj_opt = Float64[]
            list_lang_opt = Float64[]
            list_thetas_pen_opt = Float64[]
            #list_obj_sinP_opt = Float64[]

            for k in 1:T
                yⱼₖ = data_yₜ[k,:] 
                xⱼₖ = data_xₜ[k,:]
                θₖᵒᵖᵗ = θₖᵛ_dict[k]
                λₖ = λₖᵛ_dict[k] 
                objInSample = optimizacion_penalizada_2(I_cli,L,α,θ_mean,λₖ,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,yⱼₖ,xⱼₖ,pᵢ,cᵢ,θₖᵒᵖᵗ)

                #--- Sanity checks!
                #ŷⱼₜ,d = forecast(θₖᵒᵖᵗ,yⱼₖ₋₁)
                #z = planPolicy(ŷⱼₜ,Cᵢⱼ)
                #obj_sinP = costAssessment(z,yⱼₖ,Cᵢⱼ,d)

                #push!(list_obj_sinP_opt,obj_sinP)
                push!(list_thetas_pen_opt,(α/2)*(norm(θₖᵒᵖᵗ - θ_mean))^2)
                push!(list_lang_opt,sum(λₖ.*θₖᵒᵖᵗ))
                push!(list_obj_opt,objInSample)
            end

            #SANITY CHECKS
            sanity_checks(list_lang_opt,list_thetas_pen_opt,λₖᵛ_dict,T)
            #----- Guardo la data!
            #append!(df,DataFrame(i = iterations, Obj_p = mean(list_obj_opt), Obj_sp = mean(list_obj_sinP_opt), del = Float64(δ), stop = "yes", pen_1= sum(list_lang_opt),pen_2 = mean(list_thetas_pen_opt)))
            append!(df,DataFrame(i = iterations, Obj_p = mean(list_obj_opt), del = Float64(δ), stop = "yes", pen_1= sum(list_lang_opt),pen_2 = mean(list_thetas_pen_opt)))
            
            jldopen("df_iter_"*string(T)*".jld2", "w") do file
                write(file, "df", df)  
            end
            return θ_mean, mean(list_obj_opt)
            break

        else
            for k in 1:T
                new_λₖᵛ⁺¹ = λₖᵛ_dict[k] .+ α*(θₖᵛ_dict[k] .- θ_mean)
                λₖᵛ_dict[k] = new_λₖᵛ⁺¹
            end
        end

        #------ End step 6
        jldopen("df_iter_"*string(T)*".jld2", "w") do file
            write(file, "df", df)  
        end
    

    end
    return θₖᵛ_dict
end

using BilevelJuMP

function optimizacion_penalizada(I,L,α,theta_bar,wₜ,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,X,pᵢ,cᵢ)


    model = BilevelModel(Gurobi.Optimizer)
    BilevelJuMP.set_mode(model, BilevelJuMP.StrongDualityMode())
    #set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", 0.03)
    #set_optimizer_attribute(model, "SolutionLimit", 1)
    #set_optimizer_attribute(model, "TimeLimit", 3600)
    
    @variables(Upper(model),begin
        θⱼ[1:(I-1),1:(L+1)]     #Tengo los thetas de dimension Ncli x Nfeatures +1 
        ŷₜ[1:(I-1)]         #Tengo las predicciones, dimension Ncli
        up_vᵢⱼ[i = 1:(I-1),i+1]   >=0
        up_Iᵢ[1:(I-1)]
        up_Iₙ[35]>=0   
        up_zᵢ[1:(I-1)] >=0
        up_yᵢ[1:(I-1)] >=0
        up_Eᵢ[1:(I-1)]
        up_pos_Eᵢ[1:(I-1)] >=0
        up_Bᵢ[1:(I-1)]
        up_pos_Bᵢ[1:(I-1)]>=0          
    end)

    @variables(Lower(model), 
    begin
        xᵢ[1:(I-1)] >=0 
        low_vᵢⱼ[i = 1:(I-1),i+1]   >=0
        low_Iᵢ[1:(I-1)]
        low_Iₙ[35]>=0   
        low_zᵢ[1:(I-1)] >=0
        low_yᵢ[1:(I-1)] >=0
        low_Eᵢ[1:(I-1)]
        low_pos_Eᵢ[1:(I-1)] >=0
        low_Bᵢ[1:(I-1)]
        low_pos_Bᵢ[1:(I-1)]>=0    
    end ) 

    @objective(Upper(model),Min,  
    sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1))
    + sum(cᵢⱼ[i]*up_vᵢⱼ[i,i+1] + pᵢ[i]*up_zᵢ[i] +up_pos_Eᵢ[i]*cᵢ[i]  + up_pos_Bᵢ[i]*(cᵢ[i]/Qᵢ[i])  for i in 1:(I-1))
    +           sum(wₜ.*θⱼ)    #wᵗ*θⱼ
    +               (α/2).*sum( (θⱼ .- theta_bar).^2 ) #|θ - x̄|^2        
    )

    @objective(Lower(model),Min,  
    sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1))
    + sum(cᵢⱼ[i]*low_vᵢⱼ[i,i+1] + pᵢ[i]*low_zᵢ[i] +low_pos_Eᵢ[i]*cᵢ[i]  + low_pos_Bᵢ[i]*(cᵢ[i]/Qᵢ[i])  for i in 1:(I-1))  
    )


    @constraints(Upper(model),
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
        [i in 1:(I-1)], up_vᵢⱼ[i,i+1] ≤ C

        # 5 
        #   ensure that, for the depot,
        #   the quantity at the end of the period is equal to the initial bike availability and 
        #   the quantity received from the last visited station minus the quantities delivered to stations
        up_Iₙ[35] == Īₙ₀ - sum(xᵢ[i] for i in 1:(I-1)) + up_vᵢⱼ[34,35]

        #6 
        #   ensure that, at the end of the rebalancing period, the number of
        #   bikes at the depot does not exceed its capacity
        up_Iₙ[35] ≤ Īₙ₀

        #7
        # ensure that, for the first visited station, 
        #the quantity at the end of rebalancing is equal to the sum between the initial available quantity 
        #and the quantity received from the depot minus the quantities used to satisfy the demand 
        #and those bikes that are redistributed to subsequent stations on the route
        up_Iᵢ[1] == Īᵢₒ[1] + xᵢ[1] - ξ[1] - up_vᵢⱼ[1,2]
        up_zᵢ[1] + up_Iᵢ[1] >=0
        up_yᵢ[1] - up_Iᵢ[1] >=0

        #8 
        #   determine the inventory position (which can be negative or positive) at a station other than the first, 
        #   as a function of the initial inventory level, the number allocated, the number withdrawn/returned, 
        #   and the number redistributed to another station in each scenario s ∈ S
        [i in 2:(I-1)], up_Iᵢ[i] == Īᵢₒ[i] + xᵢ[i] - ξ[i] + up_vᵢⱼ[i-1,i] - up_vᵢⱼ[i,i+1] 
        [i in 2:(I-1)], up_zᵢ[i] + up_Iᵢ[i] >=0
        [i in 2:(I-1)], up_yᵢ[i] - up_Iᵢ[i] >=0

        #---
        [i in 1:(I-1)], up_Eᵢ[i] == up_yᵢ[i] - Qᵢ[i]
        [i in 1:(I-1)], up_pos_Eᵢ[i] - up_Eᵢ[i] >=0

        #---
        [i in 1:(I-1)], up_Bᵢ[i] == up_yᵢ[i] - xᵢ[i] - Īᵢₒ[i] -  up_pos_Eᵢ[i]
        [i in 1:(I-1)], up_pos_Bᵢ[i] - up_Bᵢ[i] >=0

        #Prediccion! ŷₜ = Ψ(Θ,x)
        [i in 1:(I-1)], ŷₜ[i] == sum(θⱼ[i,l].*X[l] for l in 1:(L)) .+ θⱼ[i,L+1]   #θⱼ[I-1,L+1]  
    end)


    @constraints(Lower(model),
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
        [i in 1:(I-1)], low_vᵢⱼ[i,i+1] ≤ C

        # 5 
        #   ensure that, for the depot,
        #   the quantity at the end of the period is equal to the initial bike availability and 
        #   the quantity received from the last visited station minus the quantities delivered to stations
        low_Iₙ[35] == Īₙ₀ - sum(xᵢ[i] for i in 1:(I-1)) + low_vᵢⱼ[34,35]

        #6 
        #   ensure that, at the end of the rebalancing period, the number of
        #   bikes at the depot does not exceed its capacity
        low_Iₙ[35] ≤ Īₙ₀

        #7
        # ensure that, for the first visited station, 
        #the quantity at the end of rebalancing is equal to the sum between the initial available quantity 
        #and the quantity received from the depot minus the quantities used to satisfy the demand 
        #and those bikes that are redistributed to subsequent stations on the route
        low_Iᵢ[1] == Īᵢₒ[1] + xᵢ[1] - ŷₜ[1] - low_vᵢⱼ[1,2]
        low_zᵢ[1] + low_Iᵢ[1] >=0
        low_yᵢ[1] - low_Iᵢ[1] >=0

        #8 
        #   determine the inventory position (which can be negative or positive) at a station other than the first, 
        #   as a function of the initial inventory level, the number allocated, the number withdrawn/returned, 
        #   and the number redistributed to another station in each scenario s ∈ S
        [i in 2:(I-1)], low_Iᵢ[i] == Īᵢₒ[i] + xᵢ[i] - ŷₜ[i] + low_vᵢⱼ[i-1,i] - low_vᵢⱼ[i,i+1] 
        [i in 2:(I-1)], low_zᵢ[i] + low_Iᵢ[i] >=0
        [i in 2:(I-1)], low_yᵢ[i] - low_Iᵢ[i] >=0

        #---
        [i in 1:(I-1)], low_Eᵢ[i] == low_yᵢ[i] - Qᵢ[i]
        [i in 1:(I-1)], low_pos_Eᵢ[i] - low_Eᵢ[i] >=0

        #---
        [i in 1:(I-1)], low_Bᵢ[i] == low_yᵢ[i] - xᵢ[i] - Īᵢₒ[i] -  low_pos_Eᵢ[i]
        [i in 1:(I-1)], low_pos_Bᵢ[i] - low_Bᵢ[i] >=0

    end)


    optimize!(model)
    return  value.(θⱼ),objective_value(Upper(model))

end

#=
function optimizacion_penalizada_2(I,L,α,theta_bar,wₜ,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ξ,X,pᵢ,cᵢ,θₖᵒᵖᵗ)


    model = BilevelModel(Gurobi.Optimizer)
    BilevelJuMP.set_mode(model, BilevelJuMP.StrongDualityMode())
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "TimeLimit", 3600)

    θⱼ = θₖᵒᵖᵗ
    #,BilevelJuMP.StrongDualityMode()
    @variables(Upper(model),begin
#        θⱼ[1:(I-1),1:(L+1)]     #Tengo los thetas de dimension Ncli x Nfeatures +1 
        ŷₜ[1:(I-1)]         #Tengo las predicciones, dimension Ncli
        up_vᵢⱼ[i = 1:(I-1),i+1]   >=0
        up_Iᵢ[1:(I-1)]
        up_Iₙ[35]>=0   
        up_zᵢ[1:(I-1)] >=0
        up_yᵢ[1:(I-1)] >=0
        up_Eᵢ[1:(I-1)]
        up_pos_Eᵢ[1:(I-1)] >=0
        up_Bᵢ[1:(I-1)]
        up_pos_Bᵢ[1:(I-1)]>=0          
    end)

    @variables(Lower(model), 
    begin
        xᵢ[1:(I-1)] >=0 
        low_vᵢⱼ[i = 1:(I-1),i+1]   >=0
        low_Iᵢ[1:(I-1)]
        low_Iₙ[35]>=0   
        low_zᵢ[1:(I-1)] >=0
        low_yᵢ[1:(I-1)] >=0
        low_Eᵢ[1:(I-1)]
        low_pos_Eᵢ[1:(I-1)] >=0
        low_Bᵢ[1:(I-1)]
        low_pos_Bᵢ[1:(I-1)]>=0    
    end ) 

    @objective(Upper(model),Min,  
    sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1))
    + sum(cᵢⱼ[i]*up_vᵢⱼ[i,i+1] + pᵢ[i]*up_zᵢ[i] +up_pos_Eᵢ[i]*cᵢ[i]  + up_pos_Bᵢ[i]*(cᵢ[i]/Qᵢ[i])  for i in 1:(I-1))
    +           sum(wₜ.*θⱼ)    #wᵗ*θⱼ
    +               (α/2).*sum( (θⱼ .- theta_bar).^2 ) #|θ - x̄|^2        
    )

    @objective(Lower(model),Min,  
    sum(fᵢ[i]*xᵢ[i] for i in 1:(I-1))
    + sum(cᵢⱼ[i]*low_vᵢⱼ[i,i+1] + pᵢ[i]*low_zᵢ[i] +low_pos_Eᵢ[i]*cᵢ[i]  + low_pos_Bᵢ[i]*(cᵢ[i]/Qᵢ[i])  for i in 1:(I-1))  
    )


    @constraints(Upper(model),
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
        [i in 1:(I-1)], up_vᵢⱼ[i,i+1] ≤ C

        # 5 
        #   ensure that, for the depot,
        #   the quantity at the end of the period is equal to the initial bike availability and 
        #   the quantity received from the last visited station minus the quantities delivered to stations
        up_Iₙ[35] == Īₙ₀ - sum(xᵢ[i] for i in 1:(I-1)) + up_vᵢⱼ[34,35]

        #6 
        #   ensure that, at the end of the rebalancing period, the number of
        #   bikes at the depot does not exceed its capacity
        up_Iₙ[35] ≤ Īₙ₀

        #7
        # ensure that, for the first visited station, 
        #the quantity at the end of rebalancing is equal to the sum between the initial available quantity 
        #and the quantity received from the depot minus the quantities used to satisfy the demand 
        #and those bikes that are redistributed to subsequent stations on the route
        up_Iᵢ[1] == Īᵢₒ[1] + xᵢ[1] - ξ[1] - up_vᵢⱼ[1,2]
        up_zᵢ[1] + up_Iᵢ[1] >=0
        up_yᵢ[1] - up_Iᵢ[1] >=0

        #8 
        #   determine the inventory position (which can be negative or positive) at a station other than the first, 
        #   as a function of the initial inventory level, the number allocated, the number withdrawn/returned, 
        #   and the number redistributed to another station in each scenario s ∈ S
        [i in 2:(I-1)], up_Iᵢ[i] == Īᵢₒ[i] + xᵢ[i] - ξ[i] + up_vᵢⱼ[i-1,i] - up_vᵢⱼ[i,i+1] 
        [i in 2:(I-1)], up_zᵢ[i] + up_Iᵢ[i] >=0
        [i in 2:(I-1)], up_yᵢ[i] - up_Iᵢ[i] >=0

        #---
        [i in 1:(I-1)], up_Eᵢ[i] == up_yᵢ[i] - Qᵢ[i]
        [i in 1:(I-1)], up_pos_Eᵢ[i] - up_Eᵢ[i] >=0

        #---
        [i in 1:(I-1)], up_Bᵢ[i] == up_yᵢ[i] - xᵢ[i] - Īᵢₒ[i] -  up_pos_Eᵢ[i]
        [i in 1:(I-1)], up_pos_Bᵢ[i] - up_Bᵢ[i] >=0

        #Prediccion! ŷₜ = Ψ(Θ,x)
        [i in 1:(I-1)], ŷₜ[i] == sum(θⱼ[i,l].*X[l] for l in 1:(L)) .+ θⱼ[i,L+1]   #θⱼ[I-1,L+1]  
    end)


    @constraints(Lower(model),
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
        [i in 1:(I-1)], low_vᵢⱼ[i,i+1] ≤ C

        # 5 
        #   ensure that, for the depot,
        #   the quantity at the end of the period is equal to the initial bike availability and 
        #   the quantity received from the last visited station minus the quantities delivered to stations
        low_Iₙ[35] == Īₙ₀ - sum(xᵢ[i] for i in 1:(I-1)) + low_vᵢⱼ[34,35]

        #6 
        #   ensure that, at the end of the rebalancing period, the number of
        #   bikes at the depot does not exceed its capacity
        low_Iₙ[35] ≤ Īₙ₀

        #7
        # ensure that, for the first visited station, 
        #the quantity at the end of rebalancing is equal to the sum between the initial available quantity 
        #and the quantity received from the depot minus the quantities used to satisfy the demand 
        #and those bikes that are redistributed to subsequent stations on the route
        low_Iᵢ[1] == Īᵢₒ[1] + xᵢ[1] - ŷₜ[1] - low_vᵢⱼ[1,2]
        low_zᵢ[1] + low_Iᵢ[1] >=0
        low_yᵢ[1] - low_Iᵢ[1] >=0

        #8 
        #   determine the inventory position (which can be negative or positive) at a station other than the first, 
        #   as a function of the initial inventory level, the number allocated, the number withdrawn/returned, 
        #   and the number redistributed to another station in each scenario s ∈ S
        [i in 2:(I-1)], low_Iᵢ[i] == Īᵢₒ[i] + xᵢ[i] - ŷₜ[i] + low_vᵢⱼ[i-1,i] - low_vᵢⱼ[i,i+1] 
        [i in 2:(I-1)], low_zᵢ[i] + low_Iᵢ[i] >=0
        [i in 2:(I-1)], low_yᵢ[i] - low_Iᵢ[i] >=0

        #---
        [i in 1:(I-1)], low_Eᵢ[i] == low_yᵢ[i] - Qᵢ[i]
        [i in 1:(I-1)], low_pos_Eᵢ[i] - low_Eᵢ[i] >=0

        #---
        [i in 1:(I-1)], low_Bᵢ[i] == low_yᵢ[i] - xᵢ[i] - Īᵢₒ[i] -  low_pos_Eᵢ[i]
        [i in 1:(I-1)], low_pos_Bᵢ[i] - low_Bᵢ[i] >=0

    end)


    optimize!(model)
    return  objective_value(Upper(model))

end


=#