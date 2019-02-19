'''
Created on Feb 13, 2019

@author: sirjwhite
'''

import numpy
import collections

def knn(data, targets, k, metric, inputs):


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
		nearest = []
		for i in range(k):
			nearest.append(targets[distances[i][1]])

		#Compute the bag of the classes
		bag = collections.Counter(nearest)

		#Return max count tag. Stacked brackets are to remove layers of format from counter.
		toAdd = bag.most_common(1)[0][0]

		predictions.append(toAdd)

	toReturn = numpy.array(predictions)

	return toReturn

def minkowski(x,y,p):
	total = 0
	for i in range(x.shape[0]):
		total += (x[i] - y[i]) ** p
	return total ** (1/p)

def euclidean(x,y):
	return minkowski(x,y,2)

def manhattan(x,y):
	return minkowski(x,y,1)

def illegal(data, targets, k, metric, inputs):
	from sklearn.neighbors import KNeighborsClassifier
	classifier = KNeighborsClassifier(n_neighbors=k)

	classifier.fit(data,targets)
	predictions = classifier.predict(inputs)

	return predictions