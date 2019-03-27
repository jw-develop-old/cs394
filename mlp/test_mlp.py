'''
Created on Mar 18, 2019

@author: James White
'''

import t_helper

from sklearn.datasets import load_wine
from sklearn.datasets import load_iris

# Main method. Imports data and sets up for the tests.
if __name__ == '__main__':

	print("--Importing wine data--")

	wine_data = load_wine()
	set1 = (wine_data['data'].tolist(),wine_data['target'].tolist())
	out1 = 'wine_results.csv'

#	print(type(set1[0]))
#	print(len(set1[0]))
	
#	print("--Importing/parsing scene data--")

	load_iris = load_iris()
	set2 = (load_iris['data'].tolist(),load_iris['target'].tolist())
	out2 = 'iris_results.csv'

#	print(type(set2[0]))
#	print(len(set2[0]))
	
	print("--Test 1--")
	t_helper.runTest(set1,out1)
	print("--Test 2--")
	t_helper.runTest(set2,out2)