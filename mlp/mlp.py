'''
Created on Mar 18, 2019

@author: James White
'''

import numpy as np
import collections
from sklearn.neural_network import MLPClassifier
from random import uniform
<<<<<<< HEAD
import perceptron as pc
=======
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
    return Perceptron([uniform(-1,1) for n in range(n)], sigmoid)

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def perc_train_step(p, x, t, eta=.1) :
    xx = [1] + x
    assert len(xx) == len(p.weights)
    z = p(x)
    p.weights =  [p.weights[j] + eta*(t-z)*xx[j] for j in range(len(xx))]
>>>>>>> 66747c1c821914bbbb0ee40a0f0e0fc7c4a96cd1

#Primary classification unit
class Classifier:
<<<<<<< HEAD

	# Constructor
	def __init__(self,M,data,targets):
		self.M = M
		self.data = data
		self.targets = targets

	# Build the actual model.
	def build(self):

		print(self.data)
		print(self.targets)

	# Intialize percepct to random values.
		self.pct = []
		for i in range(self.M):
			self.pct.append(pc.initialize_perceptron(len(self.data)-1))

		print("Predictions: ",self.pred(self.data),"\n")

		# Repeat until termination condition.
		for i in range(5):
			# Iteration count.
			print("iteration " + str(i))

			# Perceptron data.
			print("Perceptrons:")
			for p in self.pct:
				print(str(p))
#			print(",".join([str(self.pct[0](d[0])) for d in self.data]))
			
			# For each data point at index j
			for x, t in zip(self.data,self.targets):

				# Compute all z
				z_l = []
				for p in self.pct:
					z_l.append(pc.perc_train_step(p,x,t))

				# Compute all y
				y_vals = self.pred(self.data)

				# For each output unit, calculate the error.
				d_y = []
				for y in y_vals:
					toAdd = y*(1-y)*(t-y)
					d_y.append(toAdd)

				d_z = []
				# For each hidden unit
				for z in z_l:
					sum = 0
					for d in d_y:
						sum += y
					print(type(z))
					toAdd = z*(1-z)*sum
					d_z.append(toAdd)

				# For each output unit
				w_ylk = []
				for y in y_vals:
					
					# For each perceptron
					for index in len(d_z):

						# For each weight
						for w in self.pct[z]:
							toAdd = .1*d_z[index]*z_l[index]
							w_ylk.append(toAdd)

				# Adjusting each weight of each perceptron.
				for y in y_vals:
					
					# For each perceptron
					for index in len(d_z):

						# For each weight
						for w in self.pct[z]:
							w += .1 * d_z[index]

			print("Predictions: ",y_vals,"\n")

		return self

	# Predict values based off of trained model.
	def pred(self,inputs):
		toReturn = []

		# For each data point.
		for x in range(len(inputs)):
			sum = 0
			for p in self.pct:
				sum += p(self.data[x])
			toReturn = toReturn + [pc.step(sum)]

		return toReturn

=======
    def __init__(self,M,data,targets):
        self.M = M
        self.data = data
        self.targets = targets

    def build(self):
    	p = initialize_perceptron(3)
    	for i in range(10):
		    print("iteration " + str(i))
		    print(str(p))
		    print(",".join([str(p(d[0])) for d in self.data]))
		    for d in self.data:
		        perc_train_step(p, d[0], d[1])

    def output(self,inputs):
    	return None
>>>>>>> 66747c1c821914bbbb0ee40a0f0e0fc7c4a96cd1

# Primary mlp function signatures.
def train(M,data,targets):
	print("\nTraining ...")
	model = Classifier(M,data,targets)
	model.build()
	return model

def classify(cfr,inputs):
	print("Classifying ...")
	return cfr.pred(inputs)

def illegal_train(M,data,targets,r):
	toReturn = MLPClassifier(solver='lbfgs',
							hidden_layer_sizes=(M,3),
							random_state=r)
	toReturn.fit(data, targets)
	return toReturn

def illegal_classify(mlp,inputs):
	return mlp.predict(inputs)