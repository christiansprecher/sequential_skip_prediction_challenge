import numpy as np
import pandas as pd
import os
import sys
import argparse
import glob
import time
from tensorflow.python.client import device_lib

from models import Hybrid
import utils


def run_local_test():
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

    # model.plot_model()
    model.print_summary()

    model.fit(x_rnn_train, x_fc_train, y_train, x_rnn_valid,
        x_fc_valid, y_valid, epochs=10, verbosity=2, patience = 5)

    model.plot_training()

    model.evaluate(x_rnn_test, x_fc_test, y_test, verbosity=2)
    # model.predict(x_rnn_test, x_fc_test, write_to_file = True)


def run_on_server_simple():
    print(device_lib.list_local_devices())

    start = time.process_time()
    path = '/cluster/scratch/cspreche/spotify_challenge'

    train_path = path + '/training_set_preproc'
    test_path = path + '/test_set_preproc'
    submission_path = path + '/submissions'

    tracks_path = train_path + '/log_8_20180902_000000000000.csv'
    sessions_path = train_path + '/session_log_8_20180902_000000000000.csv'

    x_rnn, x_fc, y = utils.load_training_data_simple(tracks_path, sessions_path)

    # Generate validation set
    # s = x_rnn.shape[0]
    # shuffle_indices = np.random.permutation(np.arange(s))
    # indices_train = shuffle_indices[:int(s*0.75)]
    # indices_valid = shuffle_indices[int(s*0.75):]
    #
    # x_rnn_train = x_rnn[indices_train,:,:]
    # x_fc_train = x_fc[indices_train,:]
    # y_train = y[indices_train,:]
    #
    # x_rnn_valid = x_rnn[indices_valid,:,:]
    # x_fc_valid = x_fc[indices_valid,:]
    # y_valid = y[indices_valid,:]

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

    rnn_layer_sizes = np.array([128, 64, 32])
    dense_layer_parallel_sizes = np.array([32, 32])
    dense_layer_sequential_sizes = np.array([64, 32, 1])
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

    model.print_summary()

    model.fit(x_rnn_train, x_fc_train, y_train, x_rnn_valid,
        x_fc_valid, y_valid, epochs=1000, batch_size = 100,
        verbosity=2, patience = 10)

    model.plot_training()
    model.save_model()

    model.evaluate(x_rnn_test, x_fc_test, y_test, verbosity=2)

    end = time.process_time()
    print("Model trained, time used: %4.2f seconds" % (end-start))



    # predict_logs = sorted(glob.glob(test_path + "/log_*.csv"))
    #
    # for tracks_path in predict_logs:
    #     dir = os.path.dirname(tracks_path)
    #     file = os.path.basename(tracks_path)
    #     sessions_path = dir + '/session_' + file
    #     x_rnn, x_fc = utils.load_test_data_simple(tracks_path, sessions_path)
    #
    #     model.predict(x_rnn, x_fc, write_to_file = True,
    #     overwrite = False, path = submission_path, verbosity = 2)

    end = time.process_time()
    print("All files written, time used: %4.2f seconds" % (end-start))

def predict_on_server_simple():

    start = time.process_time()
    path = '/cluster/scratch/cspreche/spotify_challenge'

    test_path = path + '/test_set_preproc'
    submission_path = path + '/submissions'

    model = Hybrid()
    model.load_model('hybrid_45_0.2835')

    predict_logs = sorted(glob.glob(test_path + "/log_*.csv"))

    start = time.process_time()

    for tracks_path in predict_logs:
        dir = os.path.dirname(tracks_path)
        file = os.path.basename(tracks_path)
        sessions_path = dir + '/session_' + file
        x_rnn, x_fc = utils.load_test_data_simple(tracks_path, sessions_path)

        model.predict(x_rnn, x_fc, write_to_file = True,
        overwrite = False, path = submission_path, verbosity = 2)

    end = time.process_time()
    print("All files written, time used: %4.2f seconds" % (end-start))


def run_on_server_using_generator():

    start = time.process_time()
    path = '/cluster/scratch/cspreche/spotify_challenge'

    rnn_layer_sizes = np.array([128, 64, 32])
    dense_layer_parallel_sizes = np.array([32, 32])
    dense_layer_sequential_sizes = np.array([64, 32, 1])
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

    model.print_summary()

    model.fit_generator(path, epochs=10000, batch_size = 64,
    steps_per_epoch = 1000, validation_steps = 100, verbosity = 2, patience = 50,
    iterations_per_file = 50)

    model.plot_training()
    model.save_model()

    end = time.process_time()
    print("Model trained, time used: %4.2f seconds" % (end-start))



def run_local_using_generator():

    start = time.process_time()
    path = '/mnt/Storage/Documents/git/spotify_challenge/data'

    rnn_layer_sizes = np.array([32, 32, 32])
    dense_layer_parallel_sizes = np.array([32, 32])
    dense_layer_sequential_sizes = np.array([32, 32, 1])
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

    model.print_summary()

    model.fit_generator(path, epochs=20, batch_size = 64,
    steps_per_epoch = 50, validation_steps = 10, verbosity=2, patience = 50,
    iterations_per_file = 5)

    model.plot_training()
    model.save_model()

    end = time.process_time()
    print("Model trained, time used: %4.2f seconds" % (end-start))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--task', required=True)

    io_args = parser.parse_args()
    task = io_args.task

    if task == '0':
        run_local_test()

    elif task == '1':
        run_on_server_simple()

    elif task == '2':
        predict_on_server_simple()

    elif task == '3':
        run_on_server_using_generator()

    elif task == '4':
        run_local_using_generator()

    else:
        print('Choose Task')
