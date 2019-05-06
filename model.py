import tensorflow as tf
import numpy as np


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class LSTM_RNN(object):
	def __init__(self,
		sess,
		seq_length,
		data_dim,
		hidden_dim,
		output_dim,
		learning_rate,
		iterations,
		num_layer,
		keep_prob):

		self.sess = sess
		self.seq_length = seq_length
		self.data_dim = data_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.num_layer = num_layer
		self.keep_prob = keep_prob
		
		self.__build__()

	def __build__(self):
		x_train, y_train, x_eval, y_eval, test_set = LSTM_RNN.create_training_data(self)

		self.x =tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim])
		self.y =tf.placeholder(tf.float32, [None, 1])

		# def Multi_Cell():
		# 	lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)
		# 	lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
		# 	return lstm_cell
		# self.cell = tf.contrib.rnn.MultiRNNCell(
		# 	[Multi_Cell() for _ in range(self.num_layer)],
		# 	state_is_tuple =True) if self.num_layer > 1 else Multi_Cell()
		self.cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)

		self.outputs, _states = tf.nn.dynamic_rnn(self.cell, self.x, dtype = tf.float32)
		self.logits = tf.contrib.layers.fully_connected(
			self.outputs[:, -1], self.output_dim, activation_fn=None)
		self.cost = tf.reduce_sum(tf.square(self.logits - self.y))
		self.optimizer =tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
		tf.summary.scalar("cost", self.cost)


		self.target = tf.placeholder(tf.float32, [None, 1])
		self.prediction = tf.placeholder(tf.float32, [None, 1])
		self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.target - self.prediction)))
		self.summary = tf.summary.merge_all()

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		#TB
		wrtier = tf.summary.FileWriter("log2")
		wrtier.add_graph(sess.graph)
		global_step = 0

		#CP
		saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("model/")

		if checkpoint and checkpoint.model_checkpoint_path:
			try:
				saver.restore(sess, checkpoint.model_checkpoint_path)
				print("Successful", checkpoint.model_checkpoint_path)
			except:
				print("Error")
		else:
			print("Error")	

		for epoch in range(self.iterations):
		    train_cost, _, summary = sess.run([self.cost, self.optimizer,self.summary], feed_dict={
		                            self.x: x_train, self.y: y_train})

		    print("[step: {}] loss: {}".format(epoch, train_cost))

		    #TB
		    wrtier.add_summary(summary, global_step=global_step)
		    global_step +=1
	 	#CP
		saver.save(sess, "model", global_step = global_step)

		# Test step
		get_accuracy = sess.run(self.logits, feed_dict={self.x: x_eval})
		# accuray = sess.run(accuray, feed_dict={
		#     prediction: get_accuracy, target:y_eval
		#     })
		rmse_val = sess.run(self.rmse, feed_dict={
		                self.target: y_eval, self.prediction: get_accuracy})

		test = sess.run(self.logits, feed_dict={self.x:test_set})

		print("RMSE: {}".format(rmse_val))

		np_min = int(16.580)
		np_max = int(21.320)

		final = test * (np_max - np_min) +np_min
		print("test_set", final)


		#test 
		plt.plot(y_eval, label = "y_eval")
		plt.plot(get_accuracy, label = "get_accuracy")
		plt.legend()
		plt.show()



	def MinMaxScalar(data):


		print("np.min", np.min(data, 0))
		print("np.max", np.max(data, 0))

		numerator = data - np.min(data, 0)
		denominator = np.max(data, 0) - np.min(data, 0)
		return numerator/ (denominator + 1e-7)



	def create_training_data(self):

		data = np.loadtxt("panda.csv", skiprows =1, delimiter =",", unpack =True, dtype = float)
		data = np.transpose(data)
		data = data[::-1]
		# print(data)
		# data = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
		print(data.shape)
		data = LSTM_RNN.MinMaxScalar(data)
		print(data[0])

		training_size = int(len(data) * 0.7)
		train_set = data[0:training_size]
		eval_set = data[training_size - self.seq_length:]
		test_set = data[0:7]
		test_set = test_set.reshape(1,7,5)


		def build_dataset(time_series, seq_length):
			x_data = []
			y_data = []
			for i in range(0, len(time_series) - seq_length):
				_x = time_series[i:i+ seq_length, :]
				_y = time_series[i + seq_length, [-1]]
				x_data.append(_x)
				y_data.append(_y)

			return np.array(x_data), np.array(y_data)

		x_train, y_train = build_dataset(train_set, self.seq_length)
		x_eval, y_eval = build_dataset(eval_set, self.seq_length)
		return x_train, y_train, x_eval, y_eval, test_set





