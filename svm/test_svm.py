'''
Created on Apr 5, 2019

@author: James White
'''

from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
import csv
import svm
import kernel
import numpy as np
import pandas as pd
import random as r
from sklearn.model_selection import train_test_split

# Module to run tests and simulate the data.
def test(data,targets):
	# X_train, X_test, y_train, y_test = train_test_split(
	# dataset[0], dataset[1], test_size=ts,random_state=r)

	X_train = data
	X_test = data
	y_train = targets
	y_test = targets

	# Kernel function.
	k = kernel.linear

	# Train and predict values.
	classifier = svm.train(data,targets,k,C=None)
	y_pred = svm.classify(classifier, X_test)

	# Print the results.
	print("\nPredictions and targets:")
	print(y_pred)
	print(y_test.T)

	# Print percentage success.
	percent = 1 - np.mean(y_pred != y_test.T)
	print("\n{:.2f}\n".format(percent))

	return percent

def runTest(dataset,target):

	test(dataset,target)

# Main method. Imports data and sets up for the tests.
if __name__ == '__main__':

	t_d = np.array([ [1,1],[2,2],[3,3],[-1,-1],[-2,-2],[-3,-3]])
	t_t = np.array([[1,1,1,-1,-1,-1]]).T

	print(t_d.shape)
	print(t_t.shape)

	a_d = 'simple.txt'
	a_i = 'simple.txt'

	# Data and targets.
	x = t_d
	y = t_t

	# Data files.
	print("Data -- Targets:")
	for a,b in zip(x,y):
		print (a," -- ",b)

	print("--Test 1--")
	runTest(t_d,t_t)

	# and_d = np.array([ [1,1],[0,0],[0,1],[1,0] ])
	# and_t = np.array([[1,0,0,0]]).T

	# b_d = 'and_results.txt'
	# b_i = 'and_iterations.txt'

	# print("--Test 2--")
	# runTest(and_d,and_t,b_d,b_i)

	# xor_d = np.array([ [0,0],[1,1],[0,1],[1,0] ])
	# xor_t = np.array([[0,0,1,1]]).T

	# c_d = 'xor_results.txt'
	# c_i = 'xor_iterations.txt'

	# print("--Test 3--")
	# runTest(xor_d,xor_t,c_d,c_i)