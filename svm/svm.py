'''
Created on Apr 5, 2019

@author: James White
'''

import numpy

class Classifier:

	# Constructor
	def __init__(self):

		# List of support vectors.
		self.a = a

		# Intercept
		self.b = b

		# Targets that correspond to support vectors.
		self.t = t

		# Kernel matrix from x to x.
		self.k = k

	# Predictive method.
	def predict(self,inputs):
		toReturn = []

		# To be sum over all nonzero support vectors.
		for x in inputs:
			toReturn.append(sum(self.a[i]*self.t[i]*self.k(x[i],x) + self.b for i in range(len(self.a))))

		return toReturn

def train(data,targets,k,C=None):

	# Method announcement.
	print("\nTraining ...")

	# Compute kernel matrix K

	# Compute P = ttTK

	# Assemble q vector of -1s

	# Assemble A matrix of ti along diagonal

	# Assemble G matrix of -1s along diagonal.

	# Assemble h vector of 0s.

	# Compute a vector by feeding P,q,G,h,A, and b=0 into QP solver.

	# Select support vectors from a that are not zero.

	toReturn = Classifier(a,b,t,k)

	return toReturn

def classify(svm,X_test):

	# Method announcement.
	print("Classifying ...")

	return svm.predict(inputs)