'''
Created on Apr 27, 2019

@author: James White
'''

def pca(data,M):

	components = data

	return components

def transform(data,components):
	
	data_t = []
	for d in data:
		data_t.append(d[2:])

	print()
	print("Before -- After:")
	for a,b in zip(data,data_t):
		print (a," -- ",b)

	return data_t