'''
Created on Mar 18, 2019

@author: James White
'''

import numpy as np
import collections
from sklearn.neural_network import MLPClassifier
from random import uniform
import perceptron

class Perceptron :
    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = activation
    def dimension(self) :
        return len(self.weights)-1
    def __call__(self, inputs) :
        return self.activation(np.dot(self.weights, [1]+inputs))
    def __str__(self) :
        return ",".join([str(w) for w in self.weights])

def initialize_perceptron(n) :
    return Perceptron([uniform(-1,1) for n in range(n)], np.sign)

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

class Classifier:
    def __init__(self,M,data,targets):
        self.M = M
        self.data = data
        self.targets = targets
    def build(self) :
    	p = initialize_perceptron(3)
    	for i in range(10):
		    print("iteration " + str(i))
		    print(str(p))
		    print(",".join([str(p(d[0])) for d in data]))
		    for d in data:
		        perc_train_step(p, d[0], d[1])


    def output(self,inputs) :
    	return None

# Primary mlp function signatures.
def train(M,data,targets):
	print("\nTraining ...")
	model = Classifier(M,data,targets)
	model.build()
	return model

def classify(mlp,inputs):
	print("Classifying ...")
	return mpl.output(inputs)

def illegal_train(M,data,targets,r):
	toReturn = MLPClassifier(solver='lbfgs',
	                    	hidden_layer_sizes=(M,3),
	                    	random_state=r)
	toReturn.fit(data, targets)
	return toReturn

def illegal_classify(mlp,inputs):
	return mlp.predict(inputs)