import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import functions.labeled_set as ls


class Classifier:
    def __init__(self, input_dimension):
        raise NotImplementedError("Please Implement this method")

    #Permet de calculer la prediction sur x => renvoie un score
    def predict(self, x):
        raise NotImplementedError("Please Implement this method")

    #Permet d'entrainer le modele sur un ensemble de données
    def train(self, labeledSet):
        raise NotImplementedError("Please Implement this method")

    #Permet de calculer la qualité du système
    def accuracy(self, dataset):
        nb_ok = 0
        for i in range(dataset.size()):
            output = self.predict(dataset.getX(i))
            if (output * dataset.getY(i) > 0):
                nb_ok = nb_ok + 1
        acc = nb_ok / (dataset.size() * 1.0)
        return acc

def plot_frontiere(set, classifier, step=20):
    mmax = set.x.max(0)
    mmin = set.x.min(0)
    x1grid, x2grid = np.meshgrid(np.linspace(mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step))
    grid = np.hstack((x1grid.reshape(x1grid.size, 1), x2grid.reshape(x2grid.size, 1)))

    # calcul de la prediction pour chaque point de la grille
    res = np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res = res.reshape(x1grid.shape)
    # tracé des frontieres
    plt.contourf(x1grid, x2grid, res, colors=["red", "cyan"], levels=[-1000,0,1000], linewidth=2)
def plot2DSet(dataset):
    """ new version that displays the marker according to the value of the point (ie -1 or 1)
    """
    positive_count=0
    negative_count=0



    for i in range(dataset.nb_examples):

        if dataset.y[i] == -1:

            if negative_count==0:
                negative=np.array(dataset.x[i])
                negative_count+=1
            else:
                negative = np.vstack((negative, dataset.x[i]))

        elif dataset.y[i] == 1:
            if positive_count==0:
                positive=np.array(dataset.x[i])
                positive_count+=1
            else:
                positive = np.vstack((positive, dataset.x[i]))

    plt.scatter(positive[:,0],positive[:,1],marker='o')
    plt.scatter(negative[:,0],negative[:,1],marker='x')

class KernelBias3D:
    """prend des vecteur de dimension 2 et les envoie dans un espace 3D"""

    def transform(self,x):
        y=np.asarray([x[0],x[1],1])
        return y
class KernelBiasND:
    """kernel biais qui translate les données vers un espace de n dimensions"""

    def __init__(self,output_dimension):
        self.output_dimension=output_dimension

    def transform(self,x):
        tmp= [x[i] for i in range(len(x))]
        for i in range(len(x),self.output_dimension):
            tmp.append(1)
        return tmp

class PerceptronKernel(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel,vector=None):
        # ajout d'un paramètre vector optionel, pour paramétrer manuellement
        # la condition initiale de theta
        if vector==None:
            self.dimension=dimension_kernel
            self.theta=(np.random.rand(dimension_kernel)-0.5)*10
        else:
            self.theta=vector
            self.input_dimension=len(vector)
        self.learning_rate=learning_rate
        self.kernel=kernel


    #Permet de calculer la prediction sur x => renvoie un score

    def predict(self, x):
        z = np.dot(self.kernel.transform(x), self.theta)
        if z > 0:
            return +1
        else:
            return -1

    #Permet d'entrainer le modele sur un ensemble de données
    def train(self,labeledSet):
        order = np.arange(labeledSet.size())

        np.random.shuffle(order)

        for i in order:
            if (np.dot( self.theta, self.kernel.transform( labeledSet.getX(i) )) *labeledSet.getY(i)<= 0 ):
                self.theta = self.theta + self.learning_rate * labeledSet.getY(i) * self.kernel.transform( labeledSet.getX(i) );

    #Pas besoin de définir accuracy() car elle est définie dans la classe Classifier

class KernelPoly:
    def __init__(self,input_dimension):
        #input_dimension = dimension de notre ensemble initial
        self.input_dimension=input_dimension


    def transform(self,x):
        tmp= [x[i] for i in range(len(x))]
        for i in range(self.input_dimension):
            for j in range(i,self.input_dimension):
                tmp.append(x[i]*x[j])
        tmp.append(1)
        return tmp

    def output_dimension(self):
        n = self.input_dimension
        return int(1+n+ ( (n*(n+1))/2))
