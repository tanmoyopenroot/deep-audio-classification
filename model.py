import time

import numpy as np
import tensorflow as tf

from config import img_height, img_width
from config import num_labels, batch_size, num_channel, num_epoch, eval_frequency

from util import loadDataset

SEED = 66478
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("use_fp16", False, """ Train the model using fp16 """)

def dataType():
	return tf.float16 if FLAGS.use_fp16 else tf.float32

def Weights(name, shape, stddev):
	weights = tf.Variable(
		tf.truncated_normal(shape, stddev = stddev, seed = SEED, dtype = dataType()),
		name = name
	)

	return weights

def Biases(name, shape):
	biases = tf.Variable(
		tf.zeros(shape, dtype = dataType()),
		name = name
	)

	return biases

def conv2d(scope_name, x, kernel_shape, biases_shape, stddev):
	with tf.variable_scope(scope_name) as scope:
		kernel = Weights(
			"weights",
			shape = kernel_shape,
			stddev = stddev
		)
		conv = tf.nn.conv2d(x, kernel, strides = [1, 1, 1, 1], padding = "SAME")
		biases = Biases(
			"biases",
			shape = biases_shape
		)
		pre_activation = tf.nn.bias_add(conv, biases)
		relu = tf.nn.relu(pre_activation, name = scope.name)

	return relu

def maxPool2d(x):
	pool = tf.nn.max_pool(
		x,
		ksize = [1, 2, 2, 1],
		strides = [1, 2, 2, 1],
		padding = "SAME"
	)	

	return pool

def flatTensor(x):
	x_shape = x.get_shape().as_list()
	x_feature_shape =  x_shape[1] * x_shape[2] * x_shape[3]
	x_reshape = tf.reshape(x, [x_shape[0],x_feature_shape])

	return x_reshape, x_feature_shape

def fcc(scope_name, x, weight_shape, biases_shape, stddev, activation = None):
	with tf.variable_scope(scope_name) as scope:
		weights = Weights(
			"weights",
			shape = weight_shape,
			stddev = stddev 
		)

		biases = Biases(
			"biases",
			shape = biases_shape
		)

		# x_shape = x.get_shape().as_list()
		# print(x_shape)

		if activation == "relu":
			output = tf.nn.relu(tf.matmul(x, weights) + biases)
		else :
			output = tf.matmul(x, weights) + biases

	return output

def dropout(x, drop_value = 0.5):
	return tf.nn.dropout(x, drop_value, seed = SEED)

def model(x, train = False):
	convnet = conv2d("conv_1", x, [2, 2, num_channel, 32], [32], stddev = 0.1)
	pool = maxPool2d(convnet)

	convnet = conv2d("conv_2", pool, [2, 2, 32, 64], [64], stddev = 0.1)
	pool = maxPool2d(convnet)

	convnet = conv2d("conv_3", pool, [2, 2, 64, 128], [128], stddev = 0.1)
	pool = maxPool2d(convnet)

	convnet = conv2d("conv_4", pool, [2, 2, 128, 256], [256], stddev = 0.1)
	pool = maxPool2d(convnet)

	flat, feature_shape = flatTensor(pool)

	flat = fcc("fcc_1", flat, [feature_shape, 512], [512], stddev = 0.1, activation = "relu")

	# if train:
	# 	flat = dropout(flat, 0.5)

	flat = fcc("fcc_2", flat, [512, num_labels], [num_labels], stddev = 0.1)

	return flat

def errorRate(prediction, labels):
	return 100.0 - (
			100.0 * np.sum(np.argmax(prediction, 1) == labels) / prediction.shape[0]
		)

def evalData(data_x, sess):
	eval_data_size = data_x.shape[0]

	x = tf.placeholder(dataType(), shape = (eval_data_size, img_height, img_width, num_channel))

	logits = model(x, train = False)
	eval_prediction = tf.nn.softmax(logits)

	feed_dict = {
		x: data_x
	}

	prediction = sess.run(eval_prediction, feed_dict)

	return prediction

def trainModel():
	train_x, train_y, valid_x, valid_y, test_x, test_y = loadDataset()
	train_size = train_x.shape[0]

	x = tf.placeholder(dataType(), shape = (batch_size, img_height, img_width, num_channel), name = "x")
	y = tf.placeholder(tf.int64, shape = (batch_size, num_labels), name = "y")

	y_cls = tf.argmax(y, axis = 1)

	logits = model(x, train = True)
	labels = tf.cast(y, tf.int64)
	loss = tf.reduce_mean(
		# tf.nn.sparse_softmax_cross_entropy_with_logits(
		# 	labels = labels, 
		# 	logits = logits
		# )
		tf.nn.softmax_cross_entropy_with_logits(
			logits = logits,
            labels = labels
        )	
	)

	global_step = tf.Variable(0, dtype = dataType(), name = "global_step")
	learning_rate =  tf.train.exponential_decay (
		0.01,
		global_step * batch_size,
		train_size,
		0.96,
		staircase = True
	)

	optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
	# y_pred = tf.nn.softmax(logits)

	# learning_rate = tf.constant(, dtype = dataType(), name = "learning_rate")

	# optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(loss, global_step = global_step)
	y_pred = tf.nn.softmax(logits)

	y_pred_cls = tf.argmax(y_pred, axis = 1)

	correct_pred = tf.equal(y_cls, y_pred_cls)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	start_time = time.time()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		print("Starting Training...")
		for step in xrange(num_epoch):
			offset = (step *  batch_size) % (train_size - batch_size)
			batch_data = train_x[offset : (offset + batch_size), ...]
			batch_labels = train_y[offset : (offset + batch_size)]

			feed_dict_train = {
				x: batch_data,
				y: batch_labels
			}

			sess.run(optimizer, feed_dict = feed_dict_train)

			if step % eval_frequency == 0 :
				l, lr, train_acc = sess.run(
					[loss, learning_rate, accuracy], 
					feed_dict = feed_dict_train
				)

				print("Epoch : {0} / {1}".format(step, num_epoch))
				print("Minibatch Loss : {0:.3f}, Learning Rate : {1:.3f}".format(l, lr))
				print("Minibatch Accuracy : {0:.3f}".format(train_acc))

				feed_dict_valid = {
					x: valid_x,
					y: valid_y
				}

				valid_acc = sess.run(accuracy, feed_dict = feed_dict_valid)
				print("Validation Accuracy : {0:.3f}".format(valid_acc))

				# valid_pred = evalData(valid_x, sess)
				# print("Validation Error : {0:.3f}".format(errorRate(valid_pred, valid_y)))

		print("Training Complete!!!")

		feed_dict_test = {
			x: test_x,
			y: test_y
		}

		test_acc = sess.run(accuracy, feed_dict = feed_dict_test)
		print("Test Accuracy : {0:.3f}".format(test_acc))

trainModel()