py"""
def net_demand(demanda,stations_id):
    Pij_prop = pd.read_csv("proporcion_flujos_v2.csv")
    Pij_prop = Pij_prop.set_index('Unnamed: 0')
    demand_Pij = Pij_prop*demanda
    list_netos = []

    for stat_i in stations_id:
        entra = demand_Pij.loc[stat_i].sum()
        sale = demand_Pij[str(stat_i)].sum()
        neto = entra-sale
        list_netos.append(neto)

    return list_netos
"""

#demanda_total = py"""list(pd.read_csv('y_todos_features.csv')["trips"])"""
#demandas_netas = net_multiple_demands(733,I_cli,station_ids,demanda_total)
#net_multiple_demands(733,I_cli,station_ids,demanda_total)