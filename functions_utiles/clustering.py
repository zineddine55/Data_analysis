import numpy as np
import pandas as pd
import math
from datetime import datetime as dt
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def normalisation(dataframe):
    new = dict();
    for column in dataframe.columns:
        new[column]=[]
        val_min = dataframe[column].min()
        val_max = dataframe[column].max()
    
        for value in dataframe[column]:
            new[column].append( float(value - val_min) / (val_max-val_min))

    return pd.DataFrame(new)
def dist_euclidienne_vect(x,y):
    ret=0
    for i in range(len(x)):
        ret+=(x[i]-y[i])**2
    return math.sqrt(ret)

def dist_manhattan_vect(x,y):
    ret = 0
    for i in range(len(x)):
        ret += abs(x[i ]-y[i])
    return ret

def dist_vect(string,x,y):
    if string == "manhattan": return dist_manhattan_vect(x,y)
    elif string == "euclidienne": return dist_euclidienne_vect(x,y)
    else: return -1


def centroide(A):
    centre = []
    for i in A[0]:
        centre.append(0)

    for i in range(0,len(A)):
        centre += A[i]
    return centre/len(A)

def dist_groupes(string,A,B):
    if string == "manhattan": return dist_manhattan_vect(centroide(A),centroide(B))
    elif string == "euclidienne": return dist_euclidienne_vect(centroide(A),centroide(B))
    else: return -1


def initialise(A):
    ret = dict()
    for i in range(len(A)):
        ret[i] = [np.array(A[i])]
    return ret

def fusionne(string,partition,quiet=True):

    dist_min = float('inf')

    keys = list(partition.keys())

    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            dist = dist_groupes(string,partition[keys[i]],partition[keys[j]])
            if dist_min > dist:
                dist_min = dist
                fusion = (keys[i],keys[j])
    new = dict()
    if not quiet : print ("Fusion de "+str(fusion[0])+" et "+str(fusion[1])+" pour une distance de "+str(dist_min))
    for i in keys:
        if i == fusion[1] : continue
        elif i == fusion[0]:
            new[max(keys)+1] = []
            for j in partition[i]:
                new[max(keys)+1].append(np.copy(j))
            for j in partition[fusion[1]]:
                new[max(keys)+1].append(np.copy(j))
        else:
            new[i] = []
            for j in partition[i]:
                new[i].append(np.copy(j))
    return (new,fusion[0],fusion[1],dist_min)

def clustering_hierarchique(dataframe,string):
    courant = initialise(dataframe)       # clustering courant, au depart:s donnees data_2D normalisees
    M_Fusion = []                        # initialisation
    while len(courant) >=2:              # tant qu'il y a 2 groupes a fusionner
        new,k1,k2,dist_min = fusionne(string,courant)
        if(len(M_Fusion)==0):
            M_Fusion = [k1,k2,dist_min,2]
        else:
            M_Fusion = np.vstack( [M_Fusion,[k1,k2,dist_min,2] ])
        courant = new


    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)
    plt.xlabel('Exemple', fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme a partir de la matrice M_Fusion:
    scipy.cluster.hierarchy.dendrogram(
        M_Fusion,
        leaf_font_size=18.,  # taille des caracteres de l'axe des X
    )

    # Affichage du resultat obtenu:
    plt.show()

    return M_Fusion


def dist_max_groupes(string,A,B):
    dist_max = 0
    if string == "manhattan":
        for i in A[:]:
            for j in B[:]:
                dist = dist_manhattan_vect(i,j)
                if dist> dist_max: dist_max = dist
        return dist_max
    elif string == "euclidienne":
        for i in A[:]:
            for j in B[:]:
                dist = dist_euclidienne_vect(i,j)
                if dist> dist_max: dist_max = dist
        return dist_max
    else: return -1

def fusionne_max(string,partition,quiet=True):

    dist_min = float('inf')

    keys = list(partition.keys())

    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            dist = dist_max_groupes(string,partition[keys[i]],partition[keys[j]])
            if dist_min > dist:
                dist_min = dist
                fusion = (keys[i],keys[j])
    new = dict()
    if(not quiet): print ("Fusion de "+str(fusion[0])+" et "+str(fusion[1])+" pour une distance de "+str(dist_min))
    for i in keys:
        if i == fusion[1] : continue
        elif i == fusion[0]:
            new[max(keys)+1] = []
            for j in partition[i]:
                new[max(keys)+1].append(np.copy(j))
            for j in partition[fusion[1]]:
                new[max(keys)+1].append(np.copy(j))
        else:
            new[i] = []
            for j in partition[i]:
                new[i].append(np.copy(j))
    return (new,fusion[0],fusion[1],dist_min)

def clustering_hierarchique_max(dataframe,string):
    courant = initialise(dataframe)       # clustering courant, au depart:s donnees data_2D normalisees
    M_Fusion = []                        # initialisation
    while len(courant) >=2:              # tant qu'il y a 2 groupes a fusionner
        new,k1,k2,dist_min = fusionne_max(string,courant)
        if(len(M_Fusion)==0):
            M_Fusion = [k1,k2,dist_min,2]
        else:
            M_Fusion = np.vstack( [M_Fusion,[k1,k2,dist_min,2] ])
        courant = new


    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)
    plt.xlabel('Exemple', fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme a partir de la matrice M_Fusion:
    scipy.cluster.hierarchy.dendrogram(
        M_Fusion,
        leaf_font_size=18.,  # taille des caracteres de l'axe des X
    )

    # Affichage du resultat obtenu:
    plt.show()

    return M_Fusion


def clustering_hierarchique2(dataM,string,label): # dataM is a matrix
    courant = initialise(dataM)       # clustering courant, au depart:s donnees data_2D normalisees

    M_Fusion = []                        # initialisation
    while len(courant) >=2:              # tant qu'il y a 2 groupes a fusionner
        new,k1,k2,dist_min = fusionne(string,courant)
        if(len(M_Fusion)==0):
            M_Fusion = [k1,k2,dist_min,2]
        else:
            M_Fusion = np.vstack( [M_Fusion,[k1,k2,dist_min,2] ])
        courant = new


    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)
    plt.xlabel('Exemple', fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme a partir de la matrice M_Fusion:
    scipy.cluster.hierarchy.dendrogram(M_Fusion,leaf_font_size=18., labels=label)
        #leaf_font_size=18.,  # taille des caracteres de l'axe des X

    # Affichage du resultat obtenu:
    plt.show()

    return M_Fusion


def clustering_hierarchique_max2(dataM,string,label):
    courant = initialise(dataM)       # clustering courant, au depart:s donnees data_2D normalisees
    M_Fusion = []                        # initialisation
    while len(courant) >=2:              # tant qu'il y a 2 groupes a fusionner
        new,k1,k2,dist_min = fusionne_max(string,courant)
        if(len(M_Fusion)==0):
            M_Fusion = [k1,k2,dist_min,2]
        else:
            M_Fusion = np.vstack( [M_Fusion,[k1,k2,dist_min,2] ])
        courant = new


    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)
    plt.xlabel('Exemple', fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme a partir de la matrice M_Fusion:
    scipy.cluster.hierarchy.dendrogram(M_Fusion,leaf_rotation = 45,leaf_font_size=18., labels = label)

    # Affichage du resultat obtenu:
    plt.show()

    return M_Fusion
