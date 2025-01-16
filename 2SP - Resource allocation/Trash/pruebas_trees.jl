include("imports.jl")
include("data.jl")
include("LS.jl")
include("residuals.jl")
include("optimalityGap.jl")
include("parameters.jl")
include("policy.jl")
include("NM_para.jl")
include("CART.jl")
include("AD_tree.jl")

#----- Parámetros
NScenarios = [10]
J = 30
I_prods = 20
ω = 1
L = 3
σ = 5
p = 2
rep = 1
N_insample = 10000
N_outofsample = 1000*30
#------ generamos matrices para crear data.
ϕ,ζ,Σ = sampleParameters(J,σ,ω)

#------ DF para guardar data
df_final= DataFrame()
df_inSample = DataFrame()
thetas_dict = Dict()

for nRep in 1:rep
    println("**************************************")
    println("Repeticion: "*string(nRep))
    #Genero la data para el experimento
    Y,Yₒₒₛ,X,X_new =  dataGeneration(N_insample,N_outofsample,ϕ,ζ,σ,J,p,L,Σ)

    #Para cada escenario
    for t_idx in eachindex(NScenarios)
        println("Cantidad de escenarios: "*string(NScenarios[t_idx]))

        #-------------- Obtencion de data de entrenamiento
        T = NScenarios[t_idx]  

        #--- REVISAR SI LOS DATOS CORRESPONDEN!
        train_yₜ = Y[:,1:T] 
        train_xₜ =  X[1:T,:]
        
        normalized_x_train,dt = normalize(train_xₜ)
        #@show train_xₜ
        #@show normalized_x_train
        ϵⱼ,ϵₘₐₓ = compute_ϵ(normalized_x_train,L)
        Rᵢⱼ = compute_Rᵢⱼ(train_yₜ,J,I_prods,cz,qw,ρᵢ ,μᵢⱼ,T)
        D = 2
        Td = 2^(D+1) -1 
        Tᵦ = Int.(collect(1:floor(T/2))) #Branch nodes
        Tₗ =  Int.(collect(floor(T/2)+1:T)) #Leaf nodes
        Nₘᵢₙ = 2
        α = 0.001

        z_tree = AD_tree_model(T,Tₗ,Tᵦ,Rᵢⱼ,α,Nₘᵢₙ,L,ϵⱼ,ϵₘₐₓ,normalized_x_train)
        @show sum(z_tree)
        #=
        println("*** Least squares")
        θₗₛ = LS(train_yₜ,train_xₜ,J,L)
        #εin := (yi − fn(xi)),
        ls_fₙ = forecast_cesgado(θₗₛ,train_xₜ,L,J,T) # ESTE ME PERMITE CALCULAR LA PREDICCION fn(Xi)
        ls_ε = residuos(ls_fₙ,train_yₜ)
        ŷₗₛ = pointPred(X_new,θₗₛ,J)
        Zₗₛ = kannanOpt(J,I_prods,ŷₗₛ ,cz,qw,ρᵢ,μᵢⱼ)

        #---- ER - SAA - OLS
        saa_zₗₛ = SAA_kannan(T,J,I_prods,ŷₗₛ .+ls_ε ,cz,qw,ρᵢ,μᵢⱼ)
        

        #------------ CART
        X2 = train_xₜ
        y2 = transpose(train_yₜ)
        X_train, X_test, y_train, y_test = py"train_test_split"(X2, y2, test_size=0.2,random_state=42)
        tree = py"getRegressor"(X_train,y_train,X_test, y_test)
        getLeav = py"getLeav"(tree,X_new,X_train,y_train)

        #Transformo la formato del AD
        θₗₛ = LS(train_yₜ,train_xₜ,J,L)
        T_hojas = length(getLeav[1][:,:,1])
        X_hoja = reshape(getLeav[1][:,:,:],(length(getLeav[1][:,:,1]),L))
        Y_hoja = transpose(reshape(getLeav[2][:,:,:],(length(getLeav[2][:,:,1]),J)) )

        #Hago AD con los samples de las hojas
        cart_res = Optim.optimize(θ->heuristicAD_par(θ,T_hojas ,Y_hoja,X_hoja,J,I_prods,cz,qw,ρᵢ,μᵢⱼ),θₗₛ  , NelderMead(), Optim.Options(f_tol=0.0001))
        cart_θₙₘ = Optim.minimizer(cart_res)
        cart_ŷ = pointPred(X_new,cart_θₙₘ,J)
        cart_Z = kannanOpt(J,I_prods,cart_ŷ ,cz,qw,ρᵢ,μᵢⱼ)
        
        #------ cart normal
        
        normal_cart_ŷ = getLeav[3]
        normal_cart_Z = kannanOpt(J,I_prods,normal_cart_ŷ ,cz,qw,ρᵢ,μᵢⱼ)
        


        println("")
        #---- NELDER MEAD 
        println("*** Nelder Mead")
        res = Optim.optimize(θ->heuristicAD_par(θ,T,train_yₜ,train_xₜ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ),θₗₛ  , NelderMead(), Optim.Options(f_tol=0.0001))
        θₙₘ = Optim.minimizer(res)
        nm_fₙ = forecast_cesgado(θₙₘ,train_xₜ,L,J,T)
        nm_ε = residuos(nm_fₙ,train_yₜ)
        ŷₙₘ  = pointPred(X_new,θₙₘ,J)

        Zₙₘ = kannanOpt(J,I_prods,ŷₙₘ ,cz,qw,ρᵢ,μᵢⱼ)
        #---- ER - SAA - AD
        #saa_zₙₘ = SAA_kannan(T,J,I_prods,ŷₙₘ .+nm_ε ,cz,qw,ρᵢ,μᵢⱼ)
        
        


        #---------------------------

        TDₗₛ = algorithm(Zₗₛ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TDₗₛ
        cart_TD = algorithm(cart_Z,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show cart_TD
        TD_cart_normal = algorithm(normal_cart_Z,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TD_cart_normal

        saa_TDₗₛ = algorithm(saa_zₗₛ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show saa_TDₗₛ
        
        TDₙₘ = algorithm(Zₙₘ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        @show TDₙₘ
        #saa_TDₙₘ = algorithm(saa_zₙₘ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        #@show saa_TDₙₘ
        
        append!(df_final, DataFrame(T = string(T), OoS = TDₗₛ , method = "LS"))
        append!(df_final, DataFrame(T = string(T), OoS = saa_TDₗₛ , method = "RES-SAA+LS"))
        append!(df_final, DataFrame(T = string(T), OoS = TDₙₘ , method = "NM"))
        #append!(df_final, DataFrame(T = string(T), OoS = saa_TDₙₘ , method = "RES-SAA+NM"))
        append!(df_final, DataFrame(T = string(T), OoS = cart_TD , method = "CART + AD"))
        append!(df_final, DataFrame(T = string(T), OoS = TD_cart_normal , method = "CART"))


        jldopen("df_final.jld2", "w") do file
            write(file, "df", df_final)  
        end
        

        =#

    end
end
