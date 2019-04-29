'''
Created on Apr 27, 2019

@author: James White
'''

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import pca
import svm
import info_svm
import multi_test
import kernel
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from termcolor import colored

def printResults(sums,y_test,y_pred):
	# Display sums computed for each value before comparing using step function.
	print("\nSums: Blue -1 | Green 1 | Red Wrong ")
	for j in range(len(sums)):
		sum = sums[j]
		if sum > 0 and y_test[j] == 1:
			print(colored("{:.2f}".format(sum),'green'),end=' ')
		elif sum < 0 and y_test[j] == -1:
			print(colored("{:.2f}".format(sum),'blue'),end=' ')
		else:
			print(colored("{:.2f}".format(sum),'red'),end=' ')
		if j % 6 == 5:
			print()

	# Print percentage success
	percent = 1 - np.mean(y_pred != y_test.T)
	print("\n{:.2f}\n".format(percent))

# Similar to recommended data, easily seperable example data.
def testOne():
	t_d = np.array([ [2,1,4],[3,2,5],[4,3,9],[-1,-3,-2],[-2,-4,-5],[-3,-5,-7]])
	t_t = np.array([[1,1,1,-1,-1,-1]]).T

	print("--Test 1--")

	# PCA Work
	components = pca.pca(t_d,2)
	data_t = pca.transform(t_d, components)

	# Print base results.
	X_train, X_test, y_train, y_test = t_d,t_d,t_t,t_t

	classifier = svm.train(X_train,y_train,kernel.linear,C=None)
	info = svm.classify(classifier, X_test,return_sums=True)

	printResults(info[1],y_test,info[0])

	# Print transformed results.
	X_train, X_test, y_train, y_test = data_t,data_t,t_t,t_t

	classifier = svm.train(X_train,y_train,kernel.linear,C=None)
	info = svm.classify(classifier, X_test,return_sums=True)

	printResults(info[1],y_test,info[0])

if __name__ == '__main__':
	testOne()