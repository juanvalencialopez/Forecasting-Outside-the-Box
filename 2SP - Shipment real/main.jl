include("imports.jl")
include("dataGeneration.jl")
include("SAA.jl")
include("optimal_gap.jl")
include("kNN.jl")
include("LS.jl")
include("NM_para.jl")
include("policy.jl")
include("CART.jl")
include("residuals.jl")

rep = 30
train_size = [100,1000,10000] 
DistanceMatrix = transpose([0.15  1.3124  1.85  1.3124;0.50026  0.93408  1.7874  1.6039; 0.93408  0.50026  1.6039  1.7874;1.3124  0.15  1.3124  1.85; 1.6039  0.50026  0.93408  1.7874;1.7874  0.93408  0.50026  1.6039; 1.85  1.3124  0.15  1.3124;1.7874  1.6039  0.50026  0.93408; 1.6039  1.7874  0.93408  0.50026; 1.3124  1.85  1.3124  0.15;0.93408  1.7874  1.6039  0.50026;0.50026  1.6039  1.7874  0.93408])
CostMatrix = 10*DistanceMatrix
A = 2.5*[0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8; 0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8; 0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8; 0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8]
B = 7.5* [0 -1 -1; -1 0 -1; -1 -1 0; 0 -1 1; -1 0 1; -1 1 0; 0 1 -1; 1 0 -1; 1 -1 0; 0 1 1; 1 0 1; 1 1 0]
J = 12
ω = 1
L = 3
σ = 5
p = 0.5
ϕ,ζ,Σ = sampleParameters(J,σ,ω)
N_insample = 10000
N_outofsample = 1000*30

df = DataFrame()

for i in 1:rep
    println("**************************************")
    println("Repeticion: "*string(i))
    Y,Yₒₒₛ,X,X_new =  dataGeneration(N_insample,N_outofsample,ϕ,ζ,σ,J,p,L,Σ)
    
    for n in train_size
        T = n
        println("Tamaño: ",T)

        train_yₜ = Y[:,1:T] 
        train_xₜ =  X[1:T,:]

        println("*** Least squares")
        θₗₛ = LS(train_yₜ,train_xₜ,J,L)

        #------------ CART
        X2 = train_xₜ
        y2 = transpose(train_yₜ)
        X_train, X_test, y_train, y_test = py"train_test_split"(X2, y2, test_size=0.1,random_state=42)
        tree = py"getRegressor"(X_train,y_train,X_test, y_test)
        getLeav = py"getLeav"(tree,X_new,X_train,y_train)

        
        ls_fₙ = forecast_cesgado(θₗₛ,train_xₜ,L,J,T) # ESTE ME PERMITE CALCULAR LA PREDICCION fn(Xi)
        ls_ε = residuos(ls_fₙ,train_yₜ)
        ŷₗₛ = pointPred(X_new,θₗₛ,J)
        ŷₗₛ[ŷₗₛ .< 0] .= 0

        zₗₛ = bertProblem(ŷₗₛ)

        #---- ER - SAA - OLSd
        saa_zₗₛ = SAA(ŷₗₛ .+ls_ε,T,CostMatrix)

        #----------- SAA
        println("SAA")
        zₛₐₐ = SAA(train_yₜ,T,CostMatrix)
        
        #---------- KNN
        
        println("KNN")
        k = minimum((Int(round(5*(T^0.4))),T-1)) 
        yₖₙₙ = generateDemandKNN(train_xₜ,train_yₜ,k,X_new)
        zₖₙₙ = kNNOpti(yₖₙₙ,k,CostMatrix)


        T_hojas = length(getLeav[1][:,:,1])
        X_hoja = reshape(getLeav[1][:,:,:],(length(getLeav[1][:,:,1]),3))
        Y_hoja = transpose(reshape(getLeav[2][:,:,:],(length(getLeav[2][:,:,1]),12))) 

        #------ CART AD
        println("M5 + AD")
        res2 =  Optim.optimize(θ->heuristicAD_par(θ,T_hojas,Y_hoja,X_hoja,CostMatrix), θₗₛ , NelderMead(), Optim.Options(f_tol=0.0001))
        θ_cart_ad = Optim.minimizer(res2)
        y_cart_ad = sum(θ_cart_ad[:,l].*X_new[l] for l in 1:3) .+ θ_cart_ad[:,4]
        z_cart_ad = bertProblem(y_cart_ad)

        #------ CART TRADICIONAL
        println("CART")
        normal_cart_ŷ = getLeav[3]
        z_cart = bertProblem(normal_cart_ŷ)
        
        #----------NM

        println("AD")
        res =  Optim.optimize(θ->heuristicAD_par(θ,T,train_yₜ,train_xₜ,CostMatrix), θₗₛ , NelderMead(), Optim.Options(f_tol=0.0001))
        θₙₘ = Optim.minimizer(res)
        yₙₘ = sum(θₙₘ[:,l].*X_new[l] for l in 1:3) .+ θₙₘ[:,4]
        zₙₘ = bertProblem(yₙₘ)

        #--------- Optimality GAP


        TDₛₐₐ = algorithm(zₛₐₐ,Yₒₒₛ)
        append!(df,DataFrame(T=string(T) , OG = TDₛₐₐ,method = "SAA"))
        @show TDₛₐₐ

        TDₖₙₙ = algorithm(zₖₙₙ,Yₒₒₛ)
        append!(df,DataFrame(T=string(T), OG = TDₖₙₙ,method = "KNN"))
        @show TDₖₙₙ

        TDₗₛ = algorithm(zₗₛ,Yₒₒₛ)
        append!(df,DataFrame(T=string(T), OG = TDₗₛ,method = "LS"))
        @show TDₗₛ

        TDₙₘ = algorithm(zₙₘ,Yₒₒₛ)
        append!(df,DataFrame(T=string(T), OG = TDₙₘ,method = "AD"))
        @show TDₙₘ

        TD_cart = algorithm(z_cart,Yₒₒₛ)
        append!(df,DataFrame(T=string(T), OG = TD_cart,method = "CART"))
        @show TD_cart

        TD_er_saa = algorithm(saa_zₗₛ,Yₒₒₛ)
        append!(df,DataFrame(T=string(T), OG = TD_er_saa,method = "ER-SAA"))
        @show TD_er_saa

        TD_cart_ad = algorithm(z_cart_ad,Yₒₒₛ)
        append!(df,DataFrame(T=string(T), OG = TD_cart_ad,method = "M5 + AD"))
        @show TD_cart_ad

        
        jldopen("./results/df_results_"*string(p)*".jld2", "w") do file
            write(file, "df", df)  
        end
        

    end
end
