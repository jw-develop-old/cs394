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
			
			# For each data point and corresponding target.
			for x, t in zip(self.data,self.targets):

				# perc = self.pct[0]
				# z = np.dot(x,perc(x))
				# error = t - pc.sigmoid(-z)
				# sigDer = pc.sigmoid(-z) * (1 - pc.sigmoid(-z))
				# perc.weights += np.dot(x,error*sigDer)

				# Compute all z
				z_l = []
				for p in self.pct:
					z_l.append(np.dot(p.weights, [1]+self.data))

				# Compute all y
				y_vals = self.pred(self.data)

				# For each output unit, calculate the error.
				d_y = []
				for y in y_vals:
					toAdd = y*(1-y)*(t-y)
					d_y.append(toAdd)

				# Calculate derivative of activation function.
				d_z = []
				for z in z_l:
					toAdd = pc.sigmoid(z)*(1-pc.sigmoid(z))
					d_z.append(toAdd)
					
				# For each perceptron
				for index in range(len(d_z)):

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

			y_vals = self.pred(self.data)

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