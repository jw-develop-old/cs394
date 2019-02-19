'''
Created on Feb 13, 2019

@author: sirjwhite
'''

import knn
import numpy as np
import csv


from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split

def runTest(ts):

	data, inputs, targets, y_test = train_test_split(
	dataset['data'], dataset['target'], test_size=ts,random_state=0)

	k = 3
	model = knn.euclidean

#	print("\nTraining shape:", data.shape)
#	print("Testing shape:", inputs.shape)

	y_pred = knn.knn(data, targets,k,model, inputs)
	y_cheat = knn.illegal(data, targets,k,model, inputs)

	percent = 1 - np.mean(y_pred != y_test)
	print("\n","{:.2f}".format(ts))

	return percent

if __name__ == '__main__':

	datasets = [load_iris(),load_wine(),load_digits()]
	dataset = load_iris()
	val = .05

	fnames = ['test_size','percent']
	w = csv.DictWriter(open('test_results.csv', 'w', newline=''), delimiter=',',
							quotechar='|', quoting=csv.QUOTE_NONE,fieldnames=fnames)
	w.writeheader()
	for ts in np.arange(val,1,val):
		w.writerow({'test_size' : "{:.2f}".format(ts),'percent' : "{:.2f}".format(runTest(ts))})