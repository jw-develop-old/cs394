'''
Created on Apr 5, 2019

@author: James White
'''

import numpy as np
from cvxopt import matrix,solvers
import kernel

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
			sum = 0
			for i in self.a:
				sum += self.a[i]*self.t[i]*self.k(x[i],x) + self.b
			print(sum)
			toReturn.append(sum)

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
	A = np.array([t[i] for i in range(dim)])
	A = matrix(A,(1,dim),'d')

	# Assemble G matrix of -1s along diagonal.
	G = np.zeros(dims)
	for i in range(dim):
		G[i][i] = -1
	G = matrix(G,dims,'d')

	# Assemble h vector of 0s.
	h = matrix(0,(dim,1),'d')
	
	# Dummy zeroes for b.
	b = matrix(0.0)

	# print("P:\n",P)
	# print("q:\n",q)
	# print("A:\n",A)
	# print("G:\n",G)
	# print("h:\n",h)

	# Compute a vector by feeding P,q,G,h,A, and b=0 into QP solver.
	# Format: sol = solvers.qp(P,q,G,h,A,b)
	a = solvers.qp(P,q,G,h,A,b)

	# LaGrange Multipliers
	l_m = np.array(a['x'])

	# Find indices of support vector indices from a that are not zero.
	s_v = np.where(l_m > 1e-5)[0]

	print(s_v)
	print(len(s_v))

	# Compute b
	sum_1 = sum([l_m[i] * t[i] * kernel.linear(data[i],data[j]) for i in s_v])
	
	b = sum([t[j] for j in s_v]) / len(s_v)

		# # Identify (the indices of) the support vectors
		# threshold=1e-5
		# support_vectors = np.where(lagrange_multipliers>threshold)[0]
		# print(support_vectors)

		# # Compute the intercept
		# w_0 = sum([Y[j][0] - sum([lagrange_multipliers[i] * Y[i][0] * linear(X[i], X[j])
		#                           for i in support_vectors])
		#            for j in support_vectors]) / len(support_vectors)
		# print(w_0)

	# Print results.
	# print("a:\n",a)
	print("l_m:\n",l_m)
	print("s_v, support vectors:\n",s_v)
	print("b:\n",b)

	toReturn = Classifier(s_v,b,t,k)

	return toReturn

def classify(svm,inputs):

	# Method announcement.
	print("Classifying ...")

	return svm.predict(inputs)