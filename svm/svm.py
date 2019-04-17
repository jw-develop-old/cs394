'''
Created on Apr 5, 2019

@author: James White
'''

import numpy as np
from cvxopt import matrix,solvers

class Classifier:

	# Constructor
	def __init__(self,a,b,t,k):

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

def train(data,t,k,C=None):

	# Method announcement.
	print("\nTraining ...")

	dim = len(data)
	dims = (dim,dim)

	# Compute kernel matrix K
	k_matr = []
	for i in range(dim):
		k_matr.append([])
		for j in range(dim):
			k_matr[i].append(k(data[i],data[j]))
	print(k_matr)

	# Compute P = ttTK
	P = t*t.T*k_matr
	P = matrix(P,dims,'d')

	# Assemble q vector of -1s
	q = matrix(-1,(dim,1),'d')

	# Assemble A matrix of ti along diagonal
	A = np.zeros(dims)
	for i in range(dim):
		A[i][i] = t[i]
	A = matrix(A,dims,'d')

	# Assemble G matrix of -1s along diagonal.
	G = np.zeros(dims)
	for i in range(dim):
		G[i][i] = -1
	G = matrix(G,dims,'d')

	# Assemble h vector of 0s.
	h = matrix(0,(dim,1),'d')
	
	# Dummy zeroes for b.
	b = h

	# print("P:\n",P)
	# print("q:\n",q)
	# print("A:\n",A)
	# print("G:\n",G)
	# print("h:\n",h)

	# Compute a vector by feeding P,q,G,h,A, and b=0 into QP solver.
	# Format: sol = solvers.qp(P,q,G,h,A,b)
	a = solvers.qp(P,q,G,h,A,b)

	# Select support vectors from a that are not zero.
	s_v = [x for x in a['x'] if x is not 0]

	# Compute b
	

	# Print results.
	# print("a:\n",a)
	print("s_v, support vectors:\n",s_v)
	# print("b:\n",b)

	toReturn = Classifier(s_v,b,t,k)

	return toReturn

def classify(svm,inputs):

	# Method announcement.
	print("Classifying ...")

	return svm.predict(inputs)