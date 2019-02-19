'''
Created on Feb 13, 2019

@author: sirjwhite
'''

import numpy as np
import collections

def knn(data, targets, k, metric, inputs):

	print(targets)

	print(inputs.shape[0])
	predictions = [range(inputs.shape[0])]

	for c in range(inputs.shape[0]):

		#Compute all distances, Sort N points by 
		distances = []
		for i in range(data.shape[0]):
			if c != i:
				distance = metric(data[c],data[i])
				toAdd = (distance,i)
				if (distance != 0):
					distances.append(toAdd)

		#Sort N points by distance from X
		distances.sort()

		#Collect first k of these sorted points
		print(distances[0])
		nearest = distances[0][1]
	#	nearest = [range(k)]
	#	for i in range(k):
	#		nearest[i] = distances[i]
	#		print(i)
	#		print(distances[i])

		#Compute the bag of the classes
	#	bag = collections.Counter(nearest)
	#	print(nearest)
	#	for x in nearest:
	#		if x is not ():
	#			bag[targets[x[1]]] += 1

		#Return max count tag
	#	toAdd = bag.most_common(1)[0][0]
	#	print("\n",toAdd)

		predictions[c] = targets[nearest]

	print(predictions)
	print(type(predictions))
	return np.array(predictions)

def euclidean(x,y):
	total = 0
	for i in range(x.shape[0]):
		total += (x[i] - y[i]) ** 2
	return total ** .5

def manhattan(x,y):
	total = 0
	for i in range(x.shape[0]):
		total += (x[i] - y[i])
	return total

def illegal(data, targets, k, metric, inputs):
	from sklearn.neighbors import KNeighborsClassifier
	classifier = KNeighborsClassifier(n_neighbors=1)

	classifier.fit(data,targets)
	predictions = classifier.predict(inputs)

	print(predictions)
	print(type(predictions))
	return predictions