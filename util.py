import os
import pickle
from random import shuffle

import cv2
import numpy as np

from config import mp3_data_path
from config import wav_data_path
from config import spec_data_path
from config import spec_slice_path
from config import pickle_data_path 

from config import per_genre
from config import validation_ratio
from config import test_ratio

from config import spec_time

from config import 	desired_width
from config import img_height, img_width

def processData(img):
	global img_height, img_width
	img_height, img_width = img.shape
	img_data = np.asarray(img, dtype = np.uint8).reshape(img_height, img_width, 1)
	img_data = img_data / 255.

	return img_data

def imageData(path):
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	img_data = processData(img)

	return img_data

def datasetName():
	name = "{0}".format(per_genre)

	return name

def datasetInfo(train_x, train_y, valid_x, valid_y, test_x, test_y):
	print("Dataset Info")
	print("train_x : {0}".format(train_x.shape))
	print("train_y : {0}".format(train_y.shape))
	
	print("valid_x : {0}".format(valid_x.shape))
	print("valid_y : {0}".format(valid_y.shape))
	
	print("test_x : {0}".format(test_x.shape))
	print("test_y : {0}".format(test_y.shape))

def loadDataset():
	# pickle_data_path = path + "pickle/"
	dataset_name = datasetName()

	train_x = pickle.load(open("{}train_x_{}.p".format(pickle_data_path, dataset_name), "rb" ))
	train_y = pickle.load(open("{}train_y_{}.p".format(pickle_data_path, dataset_name), "rb" ))

	valid_x = pickle.load(open("{}valid_x_{}.p".format(pickle_data_path, dataset_name), "rb" ))
	valid_y = pickle.load(open("{}valid_y_{}.p".format(pickle_data_path, dataset_name), "rb" ))

	test_x = pickle.load(open("{}test_x_{}.p".format(pickle_data_path, dataset_name), "rb" ))
	test_y = pickle.load(open("{}test_y_{}.p".format(pickle_data_path, dataset_name), "rb" ))
	print("Dataset Loaded!!!")

	datasetInfo(train_x, train_y, valid_x, valid_y, test_x, test_y)

	return train_x, train_y, valid_x, valid_y, test_x, test_y


def saveDataset(train_x, train_y, valid_x, valid_y, test_x, test_y):
	# pickle_data_path = path + "pickle/"

	if not os.path.exists(pickle_data_path):
		try:
			os.makedirs(pickle_data_path)
		except Exception as e:
			raise e

	dataset_name = datasetName()

	pickle.dump(train_x, open("{}train_x_{}.p".format(pickle_data_path, dataset_name), "wb"))
	pickle.dump(train_y, open("{}train_y_{}.p".format(pickle_data_path, dataset_name), "wb"))

	pickle.dump(valid_x, open("{}valid_x_{}.p".format(pickle_data_path, dataset_name), "wb" ))
	pickle.dump(valid_y, open("{}valid_y_{}.p".format(pickle_data_path,dataset_name), "wb" ))

	pickle.dump(test_x, open("{}test_x_{}.p".format(pickle_data_path, dataset_name), "wb" ))
	pickle.dump(test_y, open("{}test_y_{}.p".format(pickle_data_path, dataset_name), "wb" ))

	print("Dataset Saved")

	datasetInfo(train_x, train_y, valid_x, valid_y, test_x, test_y)

def createDatasetFromSlices():
	global height, width
	# spec_data_path = path + "specgram/"
	# spec_slice_path = path + "slices/"
	data = []


	print("Creating Dataset")
	genres = os.listdir(spec_data_path)

	for genre in genres:
		print("Processing : {0}".format(genre))
		genre_files = [file for file in os.listdir(spec_slice_path + genre + "/") if file.endswith(".png")]
		# genre_files.sort()
		genre_files = genre_files[:per_genre]
		shuffle(genre_files)

		for file in genre_files:
			img_data = imageData(spec_slice_path + genre + "/" + file)
			label = [1. if g == genre else 0. for g in genres]
			data.append((img_data, label))

	shuffle(data)

	X, Y = zip(*data)

	num_validation = int(len(X) * validation_ratio)
	num_test = int(len(X) * test_ratio)
	num_train = len(X) - (num_validation + num_test)

	train_x = np.array(X[:num_train]).reshape([-1, img_height, img_width, 1])
	train_y = np.array(Y[:num_train])

	valid_x = np.array(X[num_train : num_train + num_validation]).reshape([-1, img_height, img_width, 1])
	valid_y = np.array(Y[num_train : num_train + num_validation])

	test_x = np.array(X[-num_test : ]).reshape([-1, img_height, img_width, 1])
	test_y = np.array(Y[-num_test : ])

	print("Dataset Created...")

	saveDataset(train_x, train_y, valid_x, valid_y, test_x, test_y)

# createDatasetFromSlices()