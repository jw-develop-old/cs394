'''
Created on Feb 13, 2019

@author: sirjwhite
'''

import knn
import imgs
import numpy as np
import pandas as pd
import csv
import random as r

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split

def runTest(ts,r,dataset):
	data, inputs, targets, y_test = train_test_split(
	dataset['data'], dataset['target'], test_size=ts,random_state=r)

	k = 2
	model = knn.euclidean

	print("\nTraining shape:", data.shape)
	print("Testing shape:", inputs.shape)

	y_pred = knn.knn(data, targets,k,model, inputs)

	percent = 1 - np.mean(y_pred != y_test)
	print("\n","{:.2f}".format(ts))

	return percent

def multiTest(dataset,file):
	val = .05

	fnames = ['test_size','percent']
	w = csv.DictWriter(open(file, 'w', newline=''), delimiter=',',
								quotechar='|', quoting=csv.QUOTE_NONE,fieldnames=fnames)
	w.writeheader()

	for ts in np.arange(val,1-val,val):
		w.writerow({'test_size' : "{:.4f}".format(ts),
					'percent' : "{:.5f}".format(
						runTest(ts,r.randrange(100),dataset))})

if __name__ == '__main__':

	set1 = load_wine()
	out1 = 'wine_results.csv'

	set2 = imgs.load_scenes()
	out2 = 'scene_results.csv'

	multiTest(set1,out1)