'''
Created on Feb 13, 2019

@author: sirjwhite
'''

import numpy as np
import collections

def knn(data, targets, k, metric, inputs,not_legit):


	predictions = []

	for c in range(inputs.shape[0]):

		#Compute all distances, Sort N points by distance.
		distances = []
		for i in range(data.shape[0]):
			distance = metric(inputs[c],data[i])
			toAdd = (distance,i)
			if (distance != 0):
				distances.append(toAdd)

		#Sort N points by distance from X
		distances.sort()

		#Collect first k of these sorted points
		print(data[c])
		print(distances[0])
		print(data[distances[1][1]])
		print(targets[distances[1][1]])
		print(distances[1])
		print(data[distances[0][1]])
		print(targets[distances[0][1]])
		print()
		nearest = distances[0][1]
		nearest = []
		for i in range(k):
			nearest.append(targets[distances[i][1]])
			print(targets[distances[i][1]])
		print()

		#Compute the bag of the classes
		bag = collections.Counter(nearest)

		#Return max count tag
		toAdd = bag.most_common(1)[0][0]
		print(toAdd)
		print(not_legit[c])
		print()

		predictions.append(toAdd)

	assert len(predictions) == inputs.shape[0]
	toReturn = np.array(predictions)

	print(len(predictions))
	print(toReturn)
	print(type(toReturn))

	return toReturn

def euclidean(x,y):
	total = 0
	for i in range(x.shape[0]):
		total += (x[i] - y[i]) ** 2
	return total ** .5

def manhattan(x,y):
	total = 0
	for i in range(x.shape[0]):
		total += abs(x[i] - y[i])
	return total

def illegal(data, targets, k, metric, inputs):
	from sklearn.neighbors import KNeighborsClassifier
	classifier = KNeighborsClassifier(n_neighbors=k)

	classifier.fit(data,targets)
	predictions = classifier.predict(inputs)

	print(len(predictions))
	print(predictions)
	print(type(predictions))
	return predictions