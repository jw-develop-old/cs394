'''
Created on Feb 13, 2019

@author: sirjwhite
'''

import knn
import numpy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    iris_dataset = load_iris()

    data, inputs, targets, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

    k = 3
    model = knn.euclidean

#    print("\ndata shape:", data.shape)
#    print("targets shape:", targets.shape)
#    print("inputs shape:", inputs.shape)

    y_pred = knn.knn(data, targets,k,model, inputs)
    y_cheat = knn.illegal(data, targets,k,model, inputs)

    print("\nTest knn score: {:.2f}".format(1 - numpy.mean(y_pred != y_test)))
    print("Illegal knn score: {:.2f}\n".format(1 - numpy.mean(y_cheat != y_test)))

    # Should be .76 with 1 neighbor.