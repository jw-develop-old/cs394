# USAGE
# python classify_images.py
# python classify_images.py --model svm

# import the necessary packages
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os

def extract_color_stats(image):
	# split the input image into its respective RGB color channels
	# and then create a feature vector with 6 values: the mean and
	# standard deviation for each of the 3 channels, respectively
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
		np.std(G), np.std(B)]

	# return our set of features
	return features

def load_scenes():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", type=str, default="3scenes",
		help="path to directory containing the '3scenes' dataset")
	ap.add_argument("-m", "--model", type=str, default="knn",
		help="type of python machine learning model to use")
	args = vars(ap.parse_args())

	# grab all image paths in the input dataset directory, initialize our
	# list of extracted features and corresponding labels
	print("[INFO] extracting image features...")
	imagePaths = paths.list_images(args["dataset"])
	data = []
	labels = []

	# loop over our input images
	for imagePath in imagePaths:
		# load the input image from disk, compute color channel
		# statistics, and then update our data list
		image = Image.open(imagePath)
		features = extract_color_stats(image)
		data.append(features)

		# extract the class label from the file path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

	# encode the labels, converting them from strings to integers
	le = LabelEncoder()
	labels = le.fit_transform(labels)

	# perform a training and testing split, using 75% of the data for
	# training and 25% for evaluation
	(trainX, testX, trainY, testY) = train_test_split(data, labels,
		test_size=0.25)

	return None