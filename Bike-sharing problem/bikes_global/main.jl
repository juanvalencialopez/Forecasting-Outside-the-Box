include("imports.jl")
include("data.jl")
include("CART.jl")
include("LS.jl")
include("policy.jl")
include("knn.jl")
include("NM_para.jl")
include("residuals.jl")
include("net_demand.jl")

#----------------- Parametros
#Estaciones id
station_ids = [39,41,42,45,46,47,48,49,50,51,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77]
#Cantidad de clientes + Depot
I_cli = 35
#Costo de alocar en un principio las bicicletas 
fᵢ = ones(34)
#costos asociados de ir de i a j
cᵢⱼ = ones(34)*2 
#Cantidad minima a alocar en estacion i en 1st stage  
x̄ᵢ = [2,4,4,4,4,5,4,6,0,6,6,0,0,2,1,1,7,8,8,5,8,9,1,1,5,3,3,2,2,7,0,0,5,5] 
#Cantidad de bicicletas inicial del
Īᵢₒ = zeros(34)
#Capacidad de las estaciones
Qᵢ = [19,15,15,15,15,19,15,19,23,19,15,23,19,15,19,23,15,27,19,19,15,15,19,27,19,23,19,19,23,15,23,19,19,27,15]

#Capacidad de biciletas del depot
Īₙ₀ = 350
#Capacidad del vehiculo 
C = 25
#Costos
cᵢ = 46 .*(1 .+ [0.3742,0.2733,0.2552,0.2095,0.2095,0.2431,0.2586,0.1339,0.1408,0.1339,0.4195,0.3509,0.2025,0.3742,0.2379,0.3001,0.677,0.289,0.2489,0.2489,0.289,0.8865,0.2436,0.2379,0.3461,0.0186,0.0186,0.3914,0.4291,0.4076,0.1408,0.2025,0.3254,0.2422,0.2552])
pᵢ = cᵢ
#Cantidad de features
L = 6
#Cantidad
n_rep = 10
df = DataFrame()
df2 = DataFrame()

for repetition in 1:n_rep
    println("--------------- "*string(repetition)*" ---------------")

    #Exporto la data y la divido
    X_train, X_test, y_train, y_test  = py"bringData"("X_processed.csv","y_processed.csv") #py"bringData"("X_clusters.csv","y_todos_features.csv")
    x2,_ = size(X_train)
    x,_ = size(X_test)

    #Least squares
    println("------ LS")
    θₗₛ = reshape(LS(y_train,X_train,I_cli-1,L),L+1)

    #Application-driven 
    res = #Optim.optimize(θ->heuristicAD_par(θ,x2,X_train,y_train,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,pᵢ,cᵢ,L,station_ids),θₗₛ  , NelderMead(), Optim.Options(f_tol=0.001 ))#f_tol=0.001 time_limit = 3600
    θₙₘ = θₗₛ #Optim.minimizer(res)

    
    #Classification and Regression Trees
    println("------ CART")
    reg = py"getRegressor"(X_train,y_train,X_test,y_test)

    #Sample average aproximation
    #Obtengo las demandas netas dado un vector de demandas (ξ[t,i])
    y_saa = net_multiple_demands(x2,I_cli,station_ids,y_train)
    z_saa,_ = bikes_sharing_SAA(x2,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,y_saa,pᵢ,cᵢ)

    #Empirical residuals - SAA
    ls_fₙ = forecast_cesgado(θₗₛ,X_train,L,I_cli-1,x2) #forecast_cesgado(θₗₛ,X_train,L,I_cli-1,x2)  
    ls_ε = residuos(ls_fₙ,y_train)
    

    list_TD_cart = Float64[]
    list_TD_ls = Float64[]
    list_TD_SAA = Float64[]
    list_TD_knn = Float64[]
    list_TD_ersaa = Float64[]
    list_TD_knn_ad = Float64[]
    list_TD_ad = Float64[]
    list_TD_cart_ad = Float64[]
    
    for id_row in 1:x


        #Obtengo el feature X = xi nuevo
        X_new_i = X_test[id_row,:]
        #Obtengo la demanda real 
        #Computo la demanda con valores "netos"
        real_demand_i = py"""net_demand"""(y_test[id_row],station_ids) 


        #Predicción: Classification and Regression Trees
        getLeav = py"getLeav"(reg,X_new_i,X_train,y_train)
        normal_cart_y = getLeav[3][1]
        #Computo la prediccion con valores "netos"
        normal_cart_y = py"""net_demand"""(normal_cart_y,station_ids)

        #Obtención de datos en hoja: Classification and Regression Trees 
        X_hoja = reshape(getLeav[1][:,:,:],(length(getLeav[1][:,:,1]),L))
        Y_hoja = reshape(getLeav[2][:,:,:],(length(getLeav[1][:,:,1])))
        x3,y3 = size(X_hoja)

        #Predicción: LS
        ŷₗₛ  =  sum(θₗₛ[l].*X_new_i[l] for l in 1:L) .+ θₗₛ[L+1]
        #Computo la prediccion con valores "netos"
        ŷₗₛ = py"""net_demand"""(ŷₗₛ,station_ids)

        #Prediccion: AD
        ŷ_ad  =  sum(θₙₘ[l].*X_new_i[l] for l in 1:L) .+ θₙₘ[L+1]
        #Computo la prediccion con valores "netos"
        ŷ_ad = py"""net_demand"""(ŷ_ad,station_ids)

        #------ CART AD
        res2 = #Optim.optimize(θ->heuristicAD_par(θ,x3,X_hoja,Y_hoja,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,pᵢ,cᵢ,L,station_ids),θₗₛ , NelderMead(), Optim.Options(f_tol=0.001 )) 
        cart_θₙₘ = θₗₛ #Optim.minimizer(res2)
        cart_ŷ_ad = sum(cart_θₙₘ[l].*X_new_i[l] for l in 1:L) .+ cart_θₙₘ[L+1]
        cart_ŷ_ad = py"""net_demand"""(cart_ŷ_ad,station_ids)

        #------ ER-SAA
        


        #Prediccion: knn
        #------ knn
        k = minimum((Int(round(5*(x2^0.4))),x2-1)) 
        y_knn,x_knn = generateDemandKNN(X_train,y_train,k,X_new_i)
        xknn,yknn = size(y_knn)
        y_knn =  net_multiple_demands(xknn,I_cli,station_ids,y_knn[:]) #py"""net_demand"""(y_knn,station_ids)

        #------- Evaluacion modelos
        z_cart,_ = bikes_sharing(I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,normal_cart_y,pᵢ,cᵢ)
        z_ls,_ = bikes_sharing(I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ŷₗₛ,pᵢ,cᵢ) 
        cart_z_ad,_ = bikes_sharing(I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,cart_ŷ_ad,pᵢ,cᵢ) 
        z_knn,_ = bikes_sharing_SAA(xknn,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,y_knn,pᵢ,cᵢ)
        z_ad,_ = bikes_sharing(I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,ŷ_ad,pᵢ,cᵢ) 
        aux = sum(θₗₛ[l].*X_new_i[l] for l in 1:L) .+ θₗₛ[L+1]
        y_err_saa = aux .+ls_ε
        y_err_saa =  net_multiple_demands(x2,I_cli,station_ids,y_err_saa[:])

        saa_zₗₛ,_ = bikes_sharing_SAA(x2,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C, y_err_saa ,pᵢ,cᵢ)

        #------- Evaluo en la FO (impacto)
        TD_cart =  bikes_sharing2(z_cart,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,real_demand_i,pᵢ,cᵢ)
        TD_ls =   bikes_sharing2(z_ls,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,real_demand_i,pᵢ,cᵢ)
        TD_ad =   bikes_sharing2(z_ad,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,real_demand_i,pᵢ,cᵢ)
        TD_cart_ad =   bikes_sharing2(cart_z_ad,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,real_demand_i,pᵢ,cᵢ)
        TD_SAA =   bikes_sharing2(z_saa,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,real_demand_i,pᵢ,cᵢ)
        TD_knn = bikes_sharing2(z_knn,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,real_demand_i,pᵢ,cᵢ)
        TD_ersaa = bikes_sharing2(saa_zₗₛ,I_cli,fᵢ,cᵢⱼ, x̄ᵢ,Īᵢₒ,Qᵢ,Īₙ₀,C,real_demand_i,pᵢ,cᵢ)
 
        println("")
        @show TD_cart
        @show TD_ls
        @show TD_cart_ad
        @show TD_ad
        @show TD_SAA
        @show TD_knn
        @show TD_ersaa

        push!(list_TD_cart,TD_cart)
        push!(list_TD_ls,TD_ls)
        push!(list_TD_SAA,TD_SAA)
        push!(list_TD_knn,TD_knn)
        push!(list_TD_ersaa,TD_ersaa)
        push!(list_TD_ad,TD_ad)
        push!(list_TD_cart_ad,TD_cart_ad)

    end

    append!(df2, DataFrame(OoS = mean(list_TD_cart) , method = "CART" ))
    append!(df2, DataFrame(OoS = mean(list_TD_ls), method = "LS" )) 
    append!(df2, DataFrame(OoS = mean(list_TD_cart_ad) , method = "M5 + AD")) 
    append!(df2, DataFrame(OoS = mean(list_TD_ad) , method = "AD"))  
    append!(df2, DataFrame(OoS = mean(list_TD_SAA)  , method = "SAA")) 
    append!(df2, DataFrame(OoS = mean(list_TD_knn) , method = "KNN")) 
    append!(df2, DataFrame(OoS = mean(list_TD_ersaa)  , method = "ER-SAA")) 
    
end

