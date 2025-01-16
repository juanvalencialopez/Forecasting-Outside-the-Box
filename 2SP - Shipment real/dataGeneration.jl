#=
#Función que genera un proceso ARMA 1 para 3 clientes.
function generateDataR1_v2(iter)
    listaData = Float64[0,0,0]
    D₀ = [10 , 9, 11]
    for i in 1:iter
        #Dₜ = 0.5.*D₀ .+ 5.0 .+ rand(3)
        Dₜ = 0.5.*D₀ .+ 5.0 .+ rand(3)
        listaData = [listaData Dₜ]
        D₀ = Dₜ
    end
    return listaData[:,2:end]
end


DistanceMatrix = transpose([0.15  1.3124  1.85  1.3124;0.50026  0.93408  1.7874  1.6039; 0.93408  0.50026  1.6039  1.7874;1.3124  0.15  1.3124  1.85; 1.6039  0.50026  0.93408  1.7874;1.7874  0.93408  0.50026  1.6039; 1.85  1.3124  0.15  1.3124;1.7874  1.6039  0.50026  0.93408; 1.6039  1.7874  0.93408  0.50026; 1.3124  1.85  1.3124  0.15;0.93408  1.7874  1.6039  0.50026;0.50026  1.6039  1.7874  0.93408])
CostMatrix = 10*DistanceMatrix
=#


function armaProcess(Nsamples)
    Θ₁ = [0.4 0.8 0.0; -1.1 -0.3 0;0 0 0]
    Θ₂ = [0 -0.8 0; -1.1 0 0;0 0 0 ]

    Φ₁ = [0.5 -0.9 0; 1.1 -0.7 0 ; 0 0 0.5]
    Φ₂ = [0 -0.5 0; -0.5 0 0; 0 0 0]

    Us = sampleUs(2)
    Uₜ₋₂ = Us[:,1]
    Uₜ₋₁ = Us[:,2]

    Xₜ₋₁ = [20,30,40]
    Xₜ₋₂ = [15,25,35]

    samplesX = Float64[]
    for i in 1:Nsamples
        Uₜ = sampleUs(1)

        Xₜ = Φ₁*Xₜ₋₁ + Φ₂*Xₜ₋₂ + Θ₁*Uₜ₋₁ + Θ₂*Uₜ₋₂ + Uₜ
        #@show Xₜ
        #println(Xₜ)
        append!(samplesX,Xₜ)
        
        Xₜ₋₂ = Xₜ₋₁
        Xₜ₋₁ = Xₜ
        Uₜ₋₂ = Uₜ₋₁
        Uₜ₋₁ = Uₜ
        
    end

    
    return transpose(reshape(samplesX,3,Nsamples))
end

function sampleUs(Nsample)
    mean = [0 ; 0; 0]
    C = [1.0 0.5 0.0; 0.5 1.2 0.5; 0.0 0.5 0.8]
    d = MvNormal(mean, C)
    x = rand(d, Nsample)
    return x
end 


function getDemand(A,B,X,Σ,δ)
    len,_ = size(X)
    Y = ones(12,len)*999999
    for i in 1:12
        for n in 1:len
            Y_aux = A[i,:]'*(X[n,:] .+ δ[i]/4) .+  ((B[i,:]'*X[n,:]).*Σ[i])
            Yᵢ = maximum((0.0,Y_aux[1])) 
            Y[i,n] = Yᵢ
        end
    end
    return Y
end

function getDemand_oos(A,B,X_new,Nsamples_ooos)

    Y = ones(12,Nsamples_ooos)*999999
    for i in 1:12
        for n in 1:Nsamples_ooos
            #Y_aux = A[i,:]'*(X_new .+ randn(1)/4) .+  ((B[i,:]'*X_new).*randn(1))
            Y_aux = A[i,:]'*(transpose(X_new) .+ randn(1)/4) .+  ((B[i,:]'*transpose(X_new)).*randn(1))
            Yᵢ = maximum((0.0,Y_aux[1])) 
            Y[i,n] = Yᵢ
        end
    end
    return Y
end



function sampleParameters(J,σ,ω)
    #Obtengo todos los parámetros para cada cliente j!
    ϕⱼ = 50 .+ 5 .*rand(Normal(0,1),J)
    ζⱼ₁ = 10 .+ rand(Uniform(-4,4),J)
    ζⱼ₂ = 5 .+ rand(Uniform(-4,4),J)
    ζⱼ₃ = 2 .+ rand(Uniform(-4,4),J)
    ζ = hcat(ζⱼ₁,ζⱼ₂,ζⱼ₃)
    #πⱼₗ = 2*(ω-1)^2 *rand(Uniform(0,1),J,L)
    #ξⱼ = rand(Normal(0,σ),J)

    corr_matrix = generateRandomCorrMat(3)
    #cov_matrix = covMatrix(corr_matrix,5,3)

    return ϕⱼ,ζ,corr_matrix#cov_matrix
end

function dataGeneration(Nsamples,Noutofsamples,ϕ,ζ,σ,J,p,L,Σ)
    
    #---- Genero los x
    #X = abs.(rand(Normal(0, σ),Nsamples,L))
    #X_hoy = abs.(rand(Normal(0, σ),L))
    
    μ = zeros(3)
    X = transpose(abs.(rand(MvNormal(μ,Σ),Nsamples)))
    X_hoy = transpose(abs.(rand(MvNormal(μ,Σ),1)))


    Y = zeros(J,Nsamples)
    Yoos = zeros(J,Noutofsamples)    

    for j in 1:J
        ϕⱼ = ϕ[j]
        ζⱼ = ζ[j,:]
        #ξⱼ = ξ[j]

        for i in 1:Nsamples
            #@show X[i,:]
            Yⱼᵢ = ϕⱼ .+ sum(ζⱼ[l].*(X[i,l]).^p for l in 1:L) .+ rand(Normal(0,σ)) #ξⱼ
            Y[j,i] = Yⱼᵢ
            #@show Yⱼᵢ
        end
       
        #data in sample
        for n in 1:Noutofsamples
            oos_ξⱼ = rand(Normal(0,σ))
            oos_Yⱼ = ϕⱼ .+ sum(ζⱼ[l].*(X_hoy[l]).^p for l in 1:L) .+ oos_ξⱼ
            Yoos[j,n] = oos_Yⱼ
        end
        
    end
    
    return Y,Yoos,X,X_hoy
end

#σ = 5
#ϕ,ζ,ξ = sampleParameters(10,σ,1)
#Y,Yₒₒₛ,X,X_new =  dataGeneration(10,10,ϕ,ζ,ξ,σ,10,1,3)


function generateRandomCorrMat(dim)

	betaparam = 2.0

	partCorr = zeros(Float64,dim,dim)
	corrMat =  Matrix{Float64}(I, dim, dim) #eye(dim)

	for k = 1:dim-1
		for i = k+1:dim
			partCorr[k,i] = ((rand(Distributions.Beta(betaparam,betaparam),1))[1] - 0.5)*2.0
			p = partCorr[k,i]
            println(".")
			for j = (k-1):-1:1
				p = p*sqrt((1-partCorr[j,i]^2)*(1-partCorr[j,k]^2)) + partCorr[j,i]*partCorr[j,k]
			end
			corrMat[k,i] = p
			corrMat[i,k] = p
		end
	end

	permut = Random.randperm(dim)
	corrMat = corrMat[permut, permut]

	return corrMat
end

function covMatrix(corrM,σ,dim)
    D = Matrix{Float64}(I, dim, dim).* σ
    cov_matrix = D*corrM*D
    return cov_matrix
end
