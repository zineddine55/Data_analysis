import numpy as np
import pandas as pd
import math
import random


class LabeledSet:

    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.nb_examples = 0

    def addExample(self,vector,label):
        if (self.nb_examples == 0):
            self.x = np.array([vector])
            self.y = np.array([label])
        else:
            self.x = np.vstack((self.x, vector))
            self.y = np.vstack((self.y, label))

        self.nb_examples = self.nb_examples + 1

    #Renvoie la dimension de l'espace d'entrée
    def getInputDimension(self):
        return self.input_dimension

    #Renvoie le nombre d'exemples dans le set
    def size(self):
        return self.nb_examples

    #Renvoie la valeur de x_i
    def getX(self, i):
        return self.x[i]


    #Renvouie la valeur de y_i
    def getY(self, i):
        return(self.y[i])

def split(labeledSet):
    """Fonction qui sépare un labeledSet en 2 labeledSet aléatoirement"""
    order = np.arange(labeledSet.size())

    np.random.shuffle(order)

    new1=LabeledSet(labeledSet.getInputDimension())
    new2=LabeledSet(labeledSet.getInputDimension())
    for i in range(1,labeledSet.size()//2):
        new1.addExample(labeledSet.getX(order[i]),labeledSet.getY(order[i]))
    for i in range(labeledSet.size()//2,labeledSet.size()):
        new2.addExample(labeledSet.getX(order[i]),labeledSet.getY(order[i]))
    return (new1,new2)
