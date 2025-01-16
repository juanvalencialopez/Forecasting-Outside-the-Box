include("imports.jl")
include("data.jl")
include("LS.jl")
include("ecdfs.jl")
include("NM_para.jl")
include("residuals.jl")
include("PH.jl")
include("policy.jl")
include("optimalityGap.jl")
include("parameters.jl")

#----- Parámetros
NScenarios = [20,80,400]
J = 30
I_prods = 20
ω = 1
L = 3
σ = 5

#cz = rand(Normal(1,0.2),I_prods)  #[11,15,12]
#μᵢⱼ = rand(Normal(1,0.2),(I_prods,J)) #1 ./(exp.(rand(Normal(0,0.05),(I,J))))
#ρᵢ =  [rand(Uniform(1e-4,1)) for i in 1:I_prods]
#qw = rand(LogNormal(0.5,0.05),J).*maximum(cz)

p = 1
rep = 50
N_insample = 10000
N_outofsample = 1000*30
#------ DF para guardar data
df_final= DataFrame()
df_inSample = DataFrame()
thetas_dict = Dict()
#------ generamos matrices para crear data.
ϕ,ζ,Σ = sampleParameters(J,σ,ω)


for nRep in 1:rep
    println("Repeticion: "*string(nRep))
    #Genero la data para el experimento
    Y,Yₒₒₛ,X,X_new =  dataGeneration(N_insample,N_outofsample,ϕ,ζ,σ,J,p,L,Σ)
    #Para cada escenario
    for t_idx in eachindex(NScenarios)
        println("Cantidad de escenarios: "*string(NScenarios[t_idx]))

        #-------------- Obtencion de data de entrenamiento
        T = NScenarios[t_idx]  

        train_yₜ = Y[:,1:T] #randn(J,T) #Y[:,1:T]
        train_xₜ =  X[1:T,:] #randn(T,L)#X[1:T,:]

        @show size(train_yₜ)
        @show size(train_xₜ)

        println("*** Least squares")
        θₗₛ = LS(train_yₜ,train_xₜ,J,L)
        #εin := (yi − fn(xi)),
        ls_fₙ = forecast_cesgado(θₗₛ,train_xₜ,L,J,T)
        ls_ε = residuos(ls_fₙ,train_yₜ)

        #plotsECDF_samples(ls_ε,1,J,T)
        Zₗₛ = kannanOpt(J,I_prods,ls_fₙ ,cz,qw,ρᵢ,μᵢⱼ)
        
        
        println("*** Nelder Mead")
        res = Optim.optimize(θ->heuristicAD_par(θ,T,train_yₜ,train_xₜ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ),θₗₛ  , NelderMead(), Optim.Options(f_tol=0.00001))
        θₙₘ = Optim.minimizer(res)
        nm_fₙ = forecast_cesgado(θₙₘ,train_xₜ,L,J,T)
        nm_ε = residuos(nm_fₙ,train_yₜ)
        Zₙₘ = kannanOpt(J,I_prods,nm_fₙ ,cz,qw,ρᵢ,μᵢⱼ)

        plotsECDF_samples(ls_ε,nm_ε,J,T)
        
        
        #=
        
        #---- progrsive hedging
        θₚₕ = PH_kannan(T,J,L,train_yₜ,train_xₜ,1,0.00001,qw,cz,ρᵢ,μᵢⱼ,I_prods)
        ph_fₙ = forecast_cesgado(θₚₕ,train_xₜ,L,J,T)
        ph_ε = residuos(ph_fₙ,train_yₜ)
        Zₚₕ = kannanOpt(J,I_prods, ph_fₙ ,cz,qw,ρᵢ,μᵢⱼ)

        #@show θₚₕ
        =#

        #Agrego thetas 
        merge!(thetas_dict,Dict(string(nRep)*"_LS" => θₗₛ))
        merge!(thetas_dict,Dict(string(nRep)*"_NM" => θₙₘ))
        #merge!(thetas_dict,string(nRep)*"_PH" => θₚₕ)



        saa_zₗₛ = SAA_kannan(T,J,I_prods,ls_fₙ .+ls_ε ,cz,qw,ρᵢ,μᵢⱼ)
        saa_zₙₘ = SAA_kannan(T,J,I_prods,nm_fₙ .+nm_ε ,cz,qw,ρᵢ,μᵢⱼ)
        #saa_zₚₕ = SAA_kannan(T,J,I_prods,ph_fₙ .+ph_ε ,cz,qw,ρᵢ,μᵢⱼ)



        
        println("*** Calculando OoS")
        #LS
        TDₗₛ = algorithm(Zₗₛ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        #NM
        TDₙₘ = algorithm(Zₙₘ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        #PH
        #TDₚₕ = algorithm(Zₚₕ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)

        #Kannnan ER-SAA (Least squares)
        saa_TDₗₛ = algorithm(saa_zₗₛ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        #Kannnan ER-SAA (NelderMead)
        saa_TDₙₘ = algorithm(saa_zₙₘ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)
        #Kannnan ER-SAA (PH)
        #saa_TDₚₕ  = algorithm(saa_zₚₕ,Yₒₒₛ,J,I_prods,cz,qw,ρᵢ,μᵢⱼ)

        @show TDₗₛ
        @show TDₙₘ
        @show saa_TDₗₛ
        @show saa_TDₙₘ
        #@show saa_TDₚₕ

        append!(df_final, DataFrame(T = string(T), OoS = TDₗₛ , method = "LS"))
        append!(df_final, DataFrame(T = string(T), OoS = TDₙₘ , method = "NM"))
        #append!(df_final, DataFrame(T = string(T), OoS = TDₚₕ , method = "PH"))
        
        append!(df_final, DataFrame(T = string(T), OoS = saa_TDₗₛ , method = "RES-SAA+LS"))
        append!(df_final, DataFrame(T = string(T), OoS = saa_TDₙₘ , method = "RES-SAA+NM"))
        #append!(df_final, DataFrame(T = string(T), OoS = saa_TDₚₕ , method = "RES-SAA+PH"))

        jldopen("df_final.jld2", "w") do file
            write(file, "df", df_final)  
        end

        jldopen("df_thetas.jld2", "w") do file
            write(file, "df", thetas_dict)  
        end   
        
    end



end

