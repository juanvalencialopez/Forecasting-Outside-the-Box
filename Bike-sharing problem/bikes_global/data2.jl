py"""
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from pandas.core.tools.datetimes import to_datetime
import random 

def csv_to_dict(path,stations_id):
    df = pd.read_csv(path)
    dicts = {}
    keys = stations_id

    for id in stations_id:
        sub_df = df[df["start_station_id"]==id]
        lista_demand_i = list(sub_df["demanda"])
        #print(lista_demand_i)
        dicts[id] = lista_demand_i
        #random.sample(list1, 3)) 
    return dicts


"""

#=
def sampling_1_station(dictionary,station_id,n_sample):
    list_sampling = dictionary[station_id]
    sample = random.sample(list_sampling, n_sample)
    #return sample

def sampling_multiple_stations(dictionary,list_station_id,n_sample):
    list_samples = []
    for i in list_station_id:
        list_sampling = dictionary[i]
        sample = random.sample(list_sampling, n_sample)
        list_samples.append(sample)
    return list_samples

=#

function sampling_1_station(dictionary,station_id,n_sample)
    list_sampling = dictionary[station_id]
    sample = rand(list_sampling,n_sample)
    return sample
end

function sampling_multiple_stations(dictionary,list_station_id,n_sample)
    list_samples = ones(length(list_station_id),n_sample)*99999

    for i in 1:length(list_station_id)
        id_station = list_station_id[i]
        sample = sampling_1_station(dictionary,id_station,n_sample)
        list_samples[i,:] = sample
    end
    
    return list_samples
end


stations_id =[41,42,45,46,47,48,49,50,51,39,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77]
dictionary = py"csv_to_dict"("demands_2.csv",stations_id)



#---- Hagamos un mini sanity para ver si se distribuyen



N_samples = [10,50,100,250,500,750,1000,1250,1500,1750,2000]
I_cli = 34 # Cantidad de clientes + Depot
fᵢ = ones(I_cli) #Costo de alocar en un principio las bicicletas
cᵢⱼ = ones(I_cli)*2  #costos asociados de ir de i a j
x̄ᵢ = zeros(I_cli) #Cantidad minima a alocar en estacion i en 1st stage 
Īᵢₒ = zeros(I_cli)
Qᵢ = 50
Īₙ₀ = 1000
C = 30

df_in = DataFrame()
df_out = DataFrame()
ξ_true = sampling_multiple_stations(dictionary,stations_id,2000)
z_true,obj_true  = bikes_sharing_SAA(2000,I_cli,fᵢ, cᵢⱼ, x̄ᵢ, Īᵢₒ, Qᵢ, Īₙ₀, C, ξ_true)

ξ = sampling_multiple_stations(dictionary,stations_id,2000) #randn(11,100000).*10
for n in N_samples
    T = n
    ξ_train = ξ[:,1:T]
    z,obj_insample = bikes_sharing_SAA(T,I_cli,fᵢ, cᵢⱼ, x̄ᵢ, Īᵢₒ, Qᵢ, Īₙ₀, C, ξ_train)
    append!(df_in,DataFrame(T = string(T) , obj = obj_insample ))

    obj_oos = bikes_sharing_SAA_oos(z,T,I_cli,fᵢ, cᵢⱼ, x̄ᵢ, Īᵢₒ, Qᵢ, Īₙ₀, C, ξ_true)
    append!(df_out,DataFrame(T = string(T), obj = obj_oos, gap = abs(obj_true-obj_oos)*100/obj_true ))
end
