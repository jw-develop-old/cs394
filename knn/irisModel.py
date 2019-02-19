'''
Created on Feb 13, 2019

@author: sirjwhite
'''

import knn
from sklearn.datasets import load_wine
import numpy as np

class irisModel:
	def __init__(self):
		iris_dataset = load_wine()

		print("Keys of iris_dataset:\n", iris_dataset.keys())

		from sklearn.model_selection import train_test_split

		X_train, X_test, y_train, y_test = train_test_split(
	    iris_dataset['data'], iris_dataset['target'], random_state=0)
		
		print("X_train shape:", X_train.shape)
		print("y_train shape:", y_train.shape)

		print("X_test shape:", X_test.shape)
		print("y_test shape:", y_test.shape)

		from sklearn.neighbors import KNeighborsClassifier
		knn = KNeighborsClassifier(n_neighbors=1)

		knn.fit(X_train, y_train)

	#	X_new = np.array([[5, 2.9, 1, 0.2]])
	#	print("X_new.shape:", X_new.shape)

	#	prediction = knn.predict(X_new)
	#	print("Prediction:", prediction)
	#	print("Predicted target name:",
    #	iris_dataset['target_names'][prediction])

		y_pred = knn.predict(X_test)
		print("Test set predictions:\n", y_pred)

		print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))