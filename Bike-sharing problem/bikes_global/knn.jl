
function generateDemandKNN(Xs,Yj_aux,k,Xfresh) #Obtengo nuevos features)
    kdtree = KDTree(transpose(float.(Xs)))
    #@show kdtree
    idxs, dists = knn(kdtree, Xfresh, k, true)
    #@show idxs
    Yj = Yj_aux[idxs,:] #Obtengo la demanda estimada gracias a KNN
    return Yj,Xs[idxs,:]
end