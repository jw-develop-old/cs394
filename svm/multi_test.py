'''
Created on Apr 25, 2019

@author: James White
'''


def run(data,n):

	# Trimming out iris objects that correspond to greater than 1 and converting data.
	targets = np.array([data['target'].tolist()[:n]]).T
	inputs = np.array(data['data'].tolist()[:n])

	# Turning 0's into -1's.
	for i in range(len(targets)):
		if targets[i][0] == 0:
			targets[i][0] = -1

	X_train, X_test, y_train, y_test = train_test_split(inputs,targets)