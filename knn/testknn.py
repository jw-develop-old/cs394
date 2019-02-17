'''
Created on Feb 13, 2019

@author: sirjwhite
'''

import knn
from sklearn.datasets import load_iris
import numpy as np

iris_dataset = load_iris()
def irisModel():
	print("fear not")
	print("Keys of iris_dataset:\n", iris_dataset.keys())
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
	print("hey")

if __name__ == '__main__':
    
    X_train = [1,3,4]
    y_train = [4,6,5]
    X_test = [2,3,4]
    
    knn.knn(X_train, y_train,knn.euclidean,5, X_test)

    irisModel()