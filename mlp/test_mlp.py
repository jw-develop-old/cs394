'''
Created on Mar 18, 2019

@author: James White
'''

from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
import mlp
import csv
import numpy as np
import pandas as pd
import random as r
from sklearn.model_selection import train_test_split

# Module to divide data and house tests called by testknn.py, delegated to knn.py.
# runTest is for the test itself, returning the percent success rate as calculated by knn.py.
def test(ts,r,data,target):
	# X_train, X_test, y_train, y_test = train_test_split(
	# dataset[0], dataset[1], test_size=ts,random_state=r)

	X_train = data
	X_test = data
	y_train = target
	y_test = target

	# Perceptron count
	p_count = 2

	# # Training the classifier.
	# illegal = mlp.illegal_train(p_count, X_train, y_train,r)

	# # # The predicting done by the actual model.
	# # y_pred = mlp.illegal_classify(illegal, X_test)

	# percent = 1 - np.mean(y_pred != y_test)
	# print("\n{:.2f}".format(percent))

	classifier = mlp.train(p_count, X_train, y_train)
	y_pred = mlp.classify(classifier, X_test)

	percent = 1 - np.mean(y_pred != y_test)
	print("\n{:.2f}\n".format(percent))

	return percent

def runTest(dataset,target, file):
	test(.2,0,data,target)

# Main method. Imports data and sets up for the tests.
if __name__ == '__main__':

# 	print("--Importing wine data--")

	gate = [([1,1],1), ([1,-1],-1), ([-1,1],-1), ([-1,-1],-1)]

	data = [[1,1],[1,-1],[-1,1],[-1,-1]]
	target = [1,-1,-1,-1]

#	wine_data = load_wine()
#	set1 = (wine_data['data'].tolist(),wine_data['target'].tolist())
	out1 = 'wine_results.csv'

##	print(type(set1[0]))
##	print(len(set1[0]))
	
##	print("--Importing/parsing scene data--")

#	load_iris = load_iris()
#	set2 = (load_iris['data'].tolist(),load_iris['target'].tolist())
# 	out2 = 'iris_results.csv'

#	print(type(set2[0]))
#	print(len(set2[0]))
	
	print("--Test 1--")
	runTest(data,target,out1)

	# print("--Test 2--")
	# runTest(set2,out2)

"""
# Calls runTest() several times at test_splits between .05 and .90.
# It also writes the data to .csv files in the directory.
def multiTest(dataset,file):
	val = .01

	# Headings and instructions for writing in.
	fnames = ['test_size','percent']
	w = csv.DictWriter(open(file, 'w', newline=''), delimiter=',',
								quotechar='|', quoting=csv.QUOTE_NONE,fieldnames=fnames)
	w.writeheader()

	#Iterative call to knn function, complete with a randomness factor.
	for ts in np.arange(.95,.98,val):
		w.writerow({'test_size' : "{:.4f}".format(ts),
					'percent' : "{:.5f}".format(
						runTest(ts,0,dataset))})
"""