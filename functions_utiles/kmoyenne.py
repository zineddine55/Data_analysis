# -*- coding: utf-8 -*-

# ------------------------------------------------
# package Projet
# UE 3I026 "IA et Data Science" -- 2017-2018
#
# Module kmoyennes.py:
# Fonctions pour le clustering
# ------------------------------------------------

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:


import numpy as np
import pandas as pd
import math
import random
from datetime import datetime as dt
import matplotlib.pyplot as plt

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(dataframe):
    new = dict();
    for column in dataframe.columns:
        new[column]=[]
        val_min = dataframe[column].min()
        val_max = dataframe[column].max()

        for value in dataframe[column]:
            new[column].append( float(value - val_min) / (val_max-val_min))
    return pd.DataFrame(new)


# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(x,y):
    sum = 0.
    for i in range(len(x)):
        sum += (x[i] - y[i])**2
    return math.sqrt(sum)

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(A):
    labels = []
    values = []
    index = 0
    for column in A:
        labels.append(column)
        values.append(0)
        tmp = A[column]
        for i in tmp:
            values[index] += i
        values[index] = values[index]/len(A)
        index+=1

    df = pd.DataFrame([tuple(values)],columns=labels)
    return df

# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(dataframe):
    centre = centroide(dataframe)
    sum = 0
    for i in range(len(dataframe)):
        sum += dist_vect(centre.iloc[0],dataframe.iloc[i])**2

    return sum

# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(K,dataframe):
    l_index = list(range(len(dataframe)))
    random.shuffle(l_index)
    labels = list(dataframe)
    values = []
    for i in range(K):
        index = l_index.pop()
        values.append(tuple(dataframe.iloc[index]))

    return pd.DataFrame(values, columns=labels)



# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(x,dataframe):
    ind_min=0
    val_min = float('inf')

    for i in range(len(dataframe)):
        dist= dist_vect(x,dataframe.iloc[i])
        if dist < val_min :
            ind_min = i
            val_min = dist

    return ind_min

# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(dataframe,centroides):
    d = dict()
    for i in range(len(centroides)):
        d[i] = []


    for i in range(len(dataframe)):
        d[ plus_proche(dataframe.iloc[i],centroides) ].append(i)
    return d

# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(dataframe,matrix):

    new_values = []

    for i in range(len(matrix)):
        new_values.append(centroide(dataframe.iloc[matrix[i]]))
    return pd.concat(new_values)

# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(dataframe,matrix):
    sum = 0
    for i in matrix.keys():
        sum += inertie_cluster(dataframe.iloc[matrix[i]])
    return sum

# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes( K, dataframe, epsilon, iter_max,quiet=False):
    centroides = initialisation(K, dataframe)
    M = affecte_cluster(dataframe, centroides)# matrice d'affectation
    J = inertie_globale(dataframe,M) # inertie globale

    J_pred = J  + epsilon +1 # inertie globale de l'itération précédante
    # initialisée de sorte a ce qu'on soit sur d'entrer dans la boucle
    i = 1
    while abs(J_pred - J) > epsilon and iter_max > 0:

        J_pred = J
        centroides = nouveaux_centroides(dataframe,M)
        M = affecte_cluster(dataframe, centroides)
        J = inertie_globale(dataframe,M)
        if(not quiet): print ("Iteration %d inertie : %.2f difference : %.2f"%(i,J,abs(J_pred-J)))
        iter_max -= 1
        i += 1
    return (centroides,M)

# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(dataframe, centroides, matrix):
    x =  centroides.columns[0]
    y = centroides.columns[1]

    color = ['blue','orangered','cyan','magenta','orange','red','green','lime','navy','indigo','pink','yellowgreen','black','brown']

    for i in range(len(matrix)):
        plt.scatter(dataframe.iloc[matrix[i]][x],dataframe.iloc[matrix[i]][y],color=color[i%len(color)])
    plt.scatter(centroides[x],centroides[y],color='r',marker='x')
# -------
