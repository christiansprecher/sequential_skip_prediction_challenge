import numpy as np
import pandas as pd
import os
import sys
import argparse

from models import Hybrid
import utils


if __name__ == '__main__':
    x_rnn, x_fc, y = utils.load_training_data_simple()

    # Generate validation set
    s = x_rnn.shape[0]
    shuffle_indices = np.random.permutation(np.arange(s))
    indices_train = shuffle_indices[:int(s*0.5)]
    indices_valid = shuffle_indices[int(s*0.5):int(s*0.75)]
    indices_test = shuffle_indices[int(s*0.75):]

    x_rnn_train = x_rnn[indices_train,:,:]
    x_fc_train = x_fc[indices_train,:]
    y_train = y[indices_train,:]

    x_rnn_valid = x_rnn[indices_valid,:,:]
    x_fc_valid = x_fc[indices_valid,:]
    y_valid = y[indices_valid,:]

    x_rnn_test = x_rnn[indices_test,:,:]
    x_fc_test = x_fc[indices_test,:]
    y_test = y[indices_test,:]

    del x_rnn, x_fc, y

    # Generate model
    rnn_layer_sizes = np.array([128, 32, 32])
    dense_layer_parallel_sizes = np.array([32, 32])
    dense_layer_sequential_sizes = np.array([32, 20, 1])
    # rnn_layer_sizes = np.array([ 32])
    # dense_layer_parallel_sizes = np.array([32])
    # dense_layer_sequential_sizes = np.array([1])
    dropout_prob_rnn = 0.3
    dropout_prob_dense = 0.3
    lambda_reg_dense = 0.001

    model = Hybrid()
    model.build_model(
        rnn_layer_sizes = rnn_layer_sizes,
        dense_layer_parallel_sizes = dense_layer_parallel_sizes,
        dense_layer_sequential_sizes = dense_layer_sequential_sizes,
        dropout_prob_rnn = dropout_prob_rnn,
        dropout_prob_dense = dropout_prob_dense,
        lambda_reg_dense = lambda_reg_dense)

    model.plot_model()
    model.print_summary()

    model.fit(x_rnn_train, x_fc_train, y_train, x_rnn_valid,
        x_fc_valid, y_valid, verbosity=2)

    model.plot_training()

    model.evaluate(x_rnn_test, x_fc_test, y_test,verbosity=2)
