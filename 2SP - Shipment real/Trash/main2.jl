#----- Importación librerias
using JuMP,Gurobi, Distributions, JLD2, LinearAlgebra, DataFrames,BilevelJuMP,Optim,JuMP,LsqFit,NearestNeighbors
const GRB_ENV = Gurobi.Env()
#----- Importacion .jls
include("dataGeneration.jl")
include("SAA.jl")
include("LS.jl")
include("kNN.jl")
include("policy.jl")
include("Loess.jl")
include("NM_para.jl")
#=
include("NM.jl")
include("PH.jl")
include("PH_hot.jl")
=#
include("outofSample.jl")

#---- Conjuntos 
dataInSample = generateDataR1_v2(10001) #load("dataIn.jld2","data") #generateDataR1_v2(10001) #Genero data In sample
dataOutSample = generateDataR1_v2(4001)[:,2002:end] #Genero data OoS. Saco los ultimos 1k para que se haya estabilizado.
NScenarios = [10,100]
#----- Parámetros
Cᵢⱼ = CostMatrix[:,1:3]
df_final = DataFrame()
Dayer = dataInSample[:,10001]
#----- Corremos rutina: PH y AD
for t_idx in eachindex(NScenarios)
    #---- Obtengo parámetros y data.
    T = NScenarios[t_idx]   
    data_yₜ = dataInSample[:,10002-T:end]  
    data_yₜ₋₁ = dataInSample[:,10001-T:10000]
    @show T
    #=
    #---- SAA
    println("--- SAA")
    zₛₐₐ = SAA(data_yₜ,T,Cᵢⱼ)


    #---- kNN
    println("--- kNN")
    k = minimum((Int(round(5*(T^0.4))),T-1)) 
    yₖₙₙ = generateDemandKNN(data_yₜ₋₁,data_yₜ,k,Dayer)
    Zₖₙₙ = kNNOpti(yₖₙₙ,k,Cᵢⱼ)

    #---- Loess
    println("--- Loess")
    βs = LOESS(data_yₜ,T,Dayer)
    Zₗₒₑₛₛ = LOESSopt(data_yₜ,T,βs,Cᵢⱼ)
    =#

    #---- LS 
    println("--- LS")
    θₗₛ = LS([0.0,0.0],data_yₜ,data_yₜ₋₁,3,T)
    yₗₛ = θₗₛ[:,1].*Dayer .+ θₗₛ[:,2]
    Zₗₛ = bertProblem(yₗₛ)

    #---- NM
    println("--- NM")
    res =  Optim.optimize(θ->heuristicAD_par(θ,T,data_yₜ,data_yₜ₋₁,Cᵢⱼ), θₗₛ , NelderMead(), Optim.Options(f_tol=0.00001))
    θₙₘ = Optim.minimizer(res)
    yₙₘ = θₙₘ[:,1].*Dayer .+ θₙₘ[:,2]
    Zₙₘ = bertProblem(yₙₘ)
    
    #---- PH cold start
    println("--- PH cold")
    α = 300
    ϵ = 0.00001
    θₚₕ = PH_bert_para(T,α,Cᵢⱼ,data_yₜ,data_yₜ₋₁,ϵ)
    yₚₕ = θₚₕ[:,1].*Dayer .+ θₚₕ[:,2]
    Zₚₕ = bertProblem(yₚₕ)
    

    #=
    #---- PH hot start
    θ2ₚₕ = PH_bert_hot(T,α,Cᵢⱼ,data_yₜ,data_yₜ₋₁,ϵ,θₙₘ)
    y2ₚₕ = θ2ₚₕ[:,1].*Dayer .+ θ2ₚₕ[:,2]
    Z2ₚₕ = bertProblem(y2ₚₕ)
    =#
    #---- valores Zs
    @show zₛₐₐ
    @show Zₗₛ
    @show Zₖₙₙ
    @show Zₗₒₑₛₛ
    @show Zₙₘ
    @show Zₚₕ
    #@show Z2ₚₕ

    #---- out of samples
    
    TDₛₐₐ, TDₗₛ, TDₖₙₙ, TDₗₒₑₛₛ, TDₙₘ, TDₚₕ, TD2ₚₕ = expectedCost(zₛₐₐ,Zₗₛ,Zₖₙₙ,Zₗₒₑₛₛ,Zₙₘ,Zₚₕ,rand(4),dataOutSample)

    #@show TDₛₐₐ
    @show TDₗₛ
    #@show TDₖₙₙ
    #@show TDₗₒₑₛₛ
    @show TDₙₘ
    @show TDₚₕ
    #@show TD2ₚₕ

    #append!(df_final, DataFrame(T = T, OoS = TDₛₐₐ , method = "SAA"))
    append!(df_final, DataFrame(T = T, OoS = TDₗₛ , method = "LS"))
    
    #append!(df_final, DataFrame(T = T, OoS = TDₖₙₙ , method = "kNN"))
    #append!(df_final, DataFrame(T = T, OoS = TDₗₒₑₛₛ , method = "loess"))
    append!(df_final, DataFrame(T = T, OoS = TDₙₘ , method = "NM"))
    append!(df_final, DataFrame(T = T, OoS = TDₚₕ , method = "PH"))
    #append!(df_final, DataFrame(T = T, OoS = TD2ₚₕ , method = "PH hot"))
    
    jldopen("df_final.jld2", "w") do file
        write(file, "df", df_final)  
    end
    
end

