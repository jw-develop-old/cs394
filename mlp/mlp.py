'''
Created on Mar 18, 2019

@author: James White
'''

import numpy as np
import collections
from sklearn.neural_network import MLPClassifier
from random import uniform
import perceptron as pc

#Primary classification unit
class Classifier:

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