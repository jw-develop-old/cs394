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

	# Select support vectors from a that are not zero.
	s_v = np.array([x for x in a['x'] if x > 1e-5])

	# Compute b

	sum_1 = sum([l_m[i] * t[i] * linear(data[i],data[j]) for i in s_v])
	
	sum([t[j] - for j in s_v]) / len(s_v)

	b = 

	# Print results.
	# print("a:\n",a)
	print("l_m:\n",s_v)
	print("s_v, support vectors:\n",s_v)
	print("b:\n",b)

	toReturn = Classifier(s_v,b,t,k)

	return toReturn

def classify(svm,inputs):

	# Method announcement.
	print("Classifying ...")

	return svm.predict(inputs)

# import numpy as np
# import cvxopt as cvxopt
# from cvxopt import solvers

# # The code that accompanies Marsland, "Machine Learning: An Algorithmic
# # Perspective" was useful in preparing this.

# # Linear kernel
# def linear(x1, x2) :
#     return 1 + np.dot(x1, x2)

# # Six points on the x axis
# X = np.array([[1.0,0.0], [2.0,0.0],[3.0,0.0],[-1.0,0.0],[-2.0,0.0],[-3.0,0.0]])

# # Points on the positive x axis are in one class, negative in the other
# Y = np.array([[1.0],[1.0],[1.0],[-1.0],[-1.0],[-1.0]])

# # Make the kernel matrix
# K = np.array([[linear(x_1, x_2) for x_2 in X] for x_1 in X])
# print("K = " + str(K))


# # This is what I think the computation of P  should be (actual matrix
# # multiplication between Y*Y.transpose() and K), but it doesn't work.
# #P = np.matmul(Y * Y.transpose(), K)
# #print(P)

# # The following apparently does component-wise (entrywise) matrix multiplication
# # (also called Hadamard product), not what is usually called
# # matrix multiplication.
# P = Y * Y.transpose() * K
# print(P)

# # Make the other  vectors and matrices
# q = -np.ones((6,1))
# print(q)

# G = -np.eye(6)
# print(G)

# h = np.zeros((6,1))
# print(h)

# A = Y.reshape(1,6)
# print(A)

# # We'll turn this into a vector in the following line when we pass
# # all these to qp
# b = 0.0

# # Solve the QP problem
# sol = cvxopt.solvers.qp(cvxopt.matrix(P),cvxopt.matrix(q),cvxopt.matrix(G),cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))

# # Retreive the lagrange multipliers
# lagrange_multipliers = np.array(sol['x'])
# print(lagrange_multipliers)

# # Identify (the indices of) the support vectors
# threshold=1e-5
# support_vectors = np.where(lagrange_multipliers>threshold)[0]
# print(support_vectors)

# # Compute the intercept
# w_0 = sum([Y[j][0] - sum([lagrange_multipliers[i] * Y[i][0] * linear(X[i], X[j])
#                           for i in support_vectors])
#            for j in support_vectors]) / len(support_vectors)
# print(w_0)

# # Function to classify datapoints (note that this relies on
# # global variables: lagrange_multipliers, Y, X, support_vectors, and w_0
# def classify(x) :
#     return np.sign(sum([lagrange_multipliers[i] * Y[i][0] * linear(X[i], x)
#                 for i in support_vectors]) + w_0)

# # Some other points, as a test set
# X_test = [[4.0, 5.0], [1.5, 0.0], [7.5, 10.0], [-5.0, 0.0], [-2.0, -4.0], [-2.0, 4.0]]

# # Classify
# for x_i in X :
#     print(str(x_i) + ": " + str(classify(x_i)))

# for x_i in X_test :
#     print(str(x_i) + ": " + str(classify(x_i)))