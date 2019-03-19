'''
Created on Mar 18, 2019

@author: James White
'''

import numpy
import collections
from sklearn.neural_network import MLPClassifier

# Primary mlp function signatures.
def train(M,data,targets):
	return None

def classify(mlp,inputs):
	return None

def illegal_train(M,data,targets,r):
	toReturn = MLPClassifier(solver='lbfgs',
	                    	hidden_layer_sizes=(M,3),
	                    	random_state=r)
	toReturn.fit(data, targets)
	return toReturn

def illegal_classify(mlp,inputs):
	return mlp.predict(inputs)