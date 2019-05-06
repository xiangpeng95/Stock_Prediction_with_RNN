import os
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from test6 import LSTM_RNN

flags = tf.app.flags
flags.DEFINE_integer("seq_length", 7, "seq_length")
flags.DEFINE_integer("data_dim", 5, "data_dim")
flags.DEFINE_integer("hidden_dim", 100, "hidden_dim")
flags.DEFINE_integer("output_dim", 1, "output_dim")
flags.DEFINE_integer("iterations" , 1000 , "iterations")
flags.DEFINE_float("learning_rate", 0.005, "learning_rate")
flags.DEFINE_integer("num_layer", 2, "num_layer")
flags.DEFINE_float("keep_prob", 0.8, "keep_prob")

FLAGS = flags.FLAGS

if not os.path.exists("logs"):
	os.mkdir("logs")


def main(_):
	
	run_cogfig = tf.ConfigProto(log_device_placement=True)
	#increase gpu-allocation moderately
	run_cogfig.gpu_options.allow_growth = True


	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	rnn_model = LSTM_RNN(
		sess,	
		seq_length = FLAGS.seq_length,
		data_dim = FLAGS.data_dim,
		hidden_dim = FLAGS.hidden_dim,
		output_dim = FLAGS.output_dim,
		iterations = FLAGS.iterations,
		learning_rate = FLAGS.learning_rate,
		num_layer = FLAGS.num_layer,
		keep_prob = FLAGS.keep_prob)



if __name__  == "__main__":
	tf.app.run()