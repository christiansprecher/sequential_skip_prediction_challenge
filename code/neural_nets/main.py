import numpy as np
import pandas as pd
import os
import sys
import argparse
import glob
import time
from tensorflow.python.client import device_lib
import datetime

from models import Hybrid, Single_RNN_Full
import utils

# Task 00:
def run_local_test():
    x_rnn, x_fc, y = utils.load_training_data_simple()

    # Generate train, valid and test set
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
    dense_layer_parallel_sizes = np.array([32])
    dense_layer_sequential_sizes = np.array([32, 20, 1])
    dropout_prob_rnn = 0.1
    dropout_prob_dense = 0.1
    lambda_reg_dense = 0.001
    lambda_reg_rnn = 0.001

    model = Hybrid()
    model.build_model(
        rnn_layer_sizes = rnn_layer_sizes,
        dense_layer_parallel_sizes = dense_layer_parallel_sizes,
        dense_layer_sequential_sizes = dense_layer_sequential_sizes,
        dropout_prob_rnn = dropout_prob_rnn,
        dropout_prob_dense = dropout_prob_dense,
        lambda_reg_dense = lambda_reg_dense,
        lambda_reg_rnn = lambda_reg_rnn,
        merge = 'maximum')

    # model = Single_RNN_Full(model_name = 'rnn_multiconcat')
    # model.build_model(
    #     rnn_layer_sizes = rnn_layer_sizes,
    #     dense_layer_sequential_sizes = dense_layer_sequential_sizes,
    #     dropout_prob_rnn = dropout_prob_rnn,
    #     dropout_prob_dense = dropout_prob_dense,
    #     lambda_reg_dense = lambda_reg_dense,
    #     lambda_reg_rnn = lambda_reg_rnn,
    #     multiple_concatenate = True)

    model.compile(optimizer = 'Adam', loss = 'm_hinge_acc')

    # model.plot_model()
    model.print_summary()

    model.fit(x_rnn_train, x_fc_train, y_train, x_rnn_valid,
        x_fc_valid, y_valid, epochs=10, verbosity=2, patience = 5)

    model.plot_training()

    model.evaluate(x_rnn_test, x_fc_test, y_test, verbosity=2)
    # model.predict(x_rnn_test, x_fc_test, write_to_file = True)

# Task 10:
def run_on_server_simple():
    print(device_lib.list_local_devices())

    start = time.process_time()
    path = '/cluster/scratch/cspreche/spotify_challenge'

    train_path = path + '/training_set_preproc'
    test_path = path + '/test_set_preproc'
    submission_path = path + '/submissions'

    tracks_path1 = train_path + '/log_8_20180902_000000000000.csv'
    sessions_path1 = train_path + '/session_log_8_20180902_000000000000.csv'

    tracks_path2 = train_path + '/log_0_20180807_000000000000.csv'
    sessions_path2 = train_path + '/session_log_0_20180807_000000000000.csv'

    tracks_path3 = train_path + '/log_6_20180801_000000000000.csv'
    sessions_path3 = train_path + '/session_log_6_20180801_000000000000.csv'

    x_rnn_train, x_fc_train, y_train = utils.load_training_data_simple(tracks_path1, sessions_path1)
    x_rnn_valid, x_fc_valid, y_valid = utils.load_training_data_simple(tracks_path2, sessions_path2)
    x_rnn_test, x_fc_test, y_test = utils.load_training_data_simple(tracks_path3, sessions_path3)

    # s = x_rnn.shape[0]
    # shuffle_indices = np.random.permutation(np.arange(s))
    # indices_train = shuffle_indices[:int(s*0.8)]
    # indices_valid = shuffle_indices[int(s*0.8):int(s*0.9)]
    # indices_test = shuffle_indices[int(s*0.9):]
    #
    # x_rnn_train = x_rnn[indices_train,:,:]
    # x_fc_train = x_fc[indices_train,:]
    # y_train = y[indices_train,:]
    #
    # x_rnn_valid = x_rnn[indices_valid,:,:]
    # x_fc_valid = x_fc[indices_valid,:]
    # y_valid = y[indices_valid,:]
    #
    # x_rnn_test = x_rnn[indices_test,:,:]
    # x_fc_test = x_fc[indices_test,:]
    # y_test = y[indices_test,:]
    #
    # del x_rnn, x_fc, y

    # rnn_layer_sizes = np.array([128, 64, 32])
    # dense_layer_parallel_sizes = np.array([32, 32])
    # dense_layer_sequential_sizes = np.array([64, 32, 1])
    rnn_layer_sizes = np.array([1024, 1024, 512, 512])
    dense_layer_parallel_sizes = np.array([512, 512])
    dense_layer_sequential_sizes = np.array([512, 64, 1])
    dropout_prob_rnn = 0.3
    dropout_prob_dense = 0.3
    lambda_reg_dense = 0.001
    lambda_reg_rnn = 0.001
    merge = 'concatenate'

    model = Hybrid()
    model.build_model(
        rnn_layer_sizes = rnn_layer_sizes,
        dense_layer_parallel_sizes = dense_layer_parallel_sizes,
        dense_layer_sequential_sizes = dense_layer_sequential_sizes,
        dropout_prob_rnn = dropout_prob_rnn,
        dropout_prob_dense = dropout_prob_dense,
        lambda_reg_dense = lambda_reg_dense,
        lambda_reg_rnn = lambda_reg_rnn,
        merge = merge)
    model.compile(optimizer = 'Adam', loss = 'm_hinge_acc', lr = 0.001)

    model.print_summary()
    # model.plot_model()

    model.fit(x_rnn_train, x_fc_train, y_train, x_rnn_valid,
        x_fc_valid, y_valid, epochs=300, batch_size = 128,
        verbosity=2, patience = 20)

    model.plot_training()
    model.save_model()

    model.evaluate(x_rnn_test, x_fc_test, y_test, verbosity=2)

    end = time.process_time()
    print("Model trained, time used: %4.2f seconds" % (end-start))

# Task 11:
def continue_on_server_simple():
    print(device_lib.list_local_devices())

    start = time.process_time()
    path = '/cluster/scratch/cspreche/spotify_challenge'

    train_path = path + '/training_set_preproc'
    test_path = path + '/test_set_preproc'
    submission_path = path + '/submissions'

    tracks_path = train_path + '/log_8_20180902_000000000000.csv'
    sessions_path = train_path + '/session_log_8_20180902_000000000000.csv'

    x_rnn, x_fc, y = utils.load_training_data_simple(tracks_path, sessions_path)

    s = x_rnn.shape[0]
    shuffle_indices = np.random.permutation(np.arange(s))
    indices_train = shuffle_indices[:int(s*0.3)]
    indices_valid = shuffle_indices[int(s*0.3):int(s*0.4)]
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

    model = Hybrid('hybrid_concatenate')
    model.load_model('hybrid_concatenate_2018-12-31_16:52')
    model.compile(optimizer = 'Adam', loss = 'm_hinge_acc', lr = 0.0001)

    model.print_summary()

    model.fit(x_rnn_train, x_fc_train, y_train, x_rnn_valid,
        x_fc_valid, y_valid, epochs=200, batch_size = 64,
        verbosity=2, patience = 40)

    model.plot_training()
    model.save_model()

    model.evaluate(x_rnn_test, x_fc_test, y_test, verbosity=2)

    end = time.process_time()
    print("Model trained, time used: %4.2f seconds" % (end-start))


# Task 12:
def predict_on_server_simple():

    start = time.process_time()
    path = '/cluster/scratch/cspreche/spotify_challenge'

    test_path = path + '/test_set_preproc'
    submission_path = path + '/submissions'

    model = Hybrid()
    model.load_model('hybrid_2019-01-02_16:36')

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

#Task 13:
def run_on_server_using_generator():

    start = time.process_time()
    path = '/cluster/scratch/cspreche/spotify_challenge'

    rnn_layer_sizes = np.array([128, 64, 32])
    dense_layer_parallel_sizes = np.array([32, 32])
    dense_layer_sequential_sizes = np.array([64, 32, 1])
    dropout_prob_rnn = 0.1
    dropout_prob_dense = 0.1
    lambda_reg_dense = 0.001
    lambda_reg_rnn = 0.001
    merge = 'concatenate'

    optimizer = 'Adam'
    lr = 0.001
    loss = 'm_hinge_acc'

    model = Hybrid()
    model.build_model(
        rnn_layer_sizes = rnn_layer_sizes,
        dense_layer_parallel_sizes = dense_layer_parallel_sizes,
        dense_layer_sequential_sizes = dense_layer_sequential_sizes,
        dropout_prob_rnn = dropout_prob_rnn,
        dropout_prob_dense = dropout_prob_dense,
        lambda_reg_rnn = lambda_reg_rnn,
        lambda_reg_dense = lambda_reg_dense,
        merge = merge)
    model.compile(optimizer = optimizer, loss = loss, lr = lr)

    model.print_summary()
    model.fit_generator(path, epochs=1000, batch_size = 128,
    steps_per_epoch = 50, validation_steps = 50, verbosity = 2, patience = 10,
    iterations_per_file = 50)
    model.plot_training()
    model.save_model()

    end = time.process_time()
    print("Model trained, time used: %4.2f seconds" % (end-start))

#Task 03:
def run_local_using_generator():

    start = time.process_time()
    path = '/mnt/Storage/Documents/git/spotify_challenge/data'

    rnn_layer_sizes = np.array([32, 32, 32])
    dense_layer_parallel_sizes = np.array([32, 32])
    dense_layer_sequential_sizes = np.array([32, 32, 1])
    dropout_prob_rnn = 0.3
    dropout_prob_dense = 0.3
    lambda_reg_dense = 0.001

    optimizer = 'Adam'
    lr = 0.0003
    loss = 'm_hinge_acc'

    model = Hybrid()
    model.build_model(
        rnn_layer_sizes = rnn_layer_sizes,
        dense_layer_parallel_sizes = dense_layer_parallel_sizes,
        dense_layer_sequential_sizes = dense_layer_sequential_sizes,
        dropout_prob_rnn = dropout_prob_rnn,
        dropout_prob_dense = dropout_prob_dense,
        lambda_reg_dense = lambda_reg_dense)
    model.compile(optimizer = optimizer, loss = loss, lr = lr)

    model.print_summary()

    model.fit_generator(path, epochs=20, batch_size = 64,
    steps_per_epoch = 250, validation_steps = 250, verbosity=2, patience = 10,
    iterations_per_file = 250)

    model.plot_training()
    model.save_model()

    end = time.process_time()
    print("Model trained, time used: %4.2f seconds" % (end-start))

def grid_search(x_rnn_train, x_fc_train, y_train, x_rnn_valid,
        x_fc_valid, y_valid, x_rnn_test, x_fc_test, y_test,
        path, batch_size = 64, epochs = 80):

    rnn_layer_sizes = np.array([32, 32, 16])
    dense_layer_parallel_sizes = np.array([32, 32, 16])
    dense_layer_sequential_sizes = np.array([32, 32, 1])
    dropout_prob_rnn = 0.1
    dropout_prob_dense = 0.1
    lambda_reg_dense = 0.001
    lambda_reg_rnn = 0.001

    # lr_range = [0.1, 0.01, 0.001, 0.0001]
    # optimizer_range = ['SGD', 'Adam']
    lr_range = [0.001]
    optimizer_range = ['Adam']
    loss_range = ['s_hinge', 'm_hinge_acc','log_loss','m_log_acc']
    merge_range = ['multiply', 'add', 'concatenate', 'maximum', 'RNN_Single', 'RNN_Concat']

    model_nr = ( len(lr_range) * len(optimizer_range) * len(loss_range)
        * len(merge_range))
    model_idx = 0

    for lr in lr_range:
        for optimizer in optimizer_range:
            for loss in loss_range:
                for merge in merge_range:

                    model = []
                    model_idx = model_idx + 1
                    print('Model %u out of %u' % (model_idx,model_nr))

                    if merge == 'RNN_Single':
                        model = Single_RNN_Full(model_name = 'rnn_single')
                        model.build_model(
                            rnn_layer_sizes = rnn_layer_sizes,
                            dense_layer_sequential_sizes = dense_layer_sequential_sizes,
                            dropout_prob_rnn = dropout_prob_rnn,
                            dropout_prob_dense = dropout_prob_dense,
                            lambda_reg_dense = lambda_reg_dense,
                            lambda_reg_rnn = lambda_reg_rnn,
                            multiple_concatenate = False)

                    elif merge == 'RNN_Concat':
                        model = Single_RNN_Full(model_name = 'rnn_concat')
                        model.build_model(
                            rnn_layer_sizes = rnn_layer_sizes,
                            dense_layer_sequential_sizes = dense_layer_sequential_sizes,
                            dropout_prob_rnn = dropout_prob_rnn,
                            dropout_prob_dense = dropout_prob_dense,
                            lambda_reg_dense = lambda_reg_dense,
                            lambda_reg_rnn = lambda_reg_rnn,
                            multiple_concatenate = True)

                    else:
                        model = Hybrid(model_name = 'hybrid_' + merge)
                        model.build_model(
                            rnn_layer_sizes = rnn_layer_sizes,
                            dense_layer_parallel_sizes = dense_layer_parallel_sizes,
                            dense_layer_sequential_sizes = dense_layer_sequential_sizes,
                            dropout_prob_rnn = dropout_prob_rnn,
                            dropout_prob_dense = dropout_prob_dense,
                            lambda_reg_dense = lambda_reg_dense,
                            lambda_reg_rnn = lambda_reg_rnn,
                            merge = merge)

                    model.compile(optimizer = optimizer, loss = loss, lr = lr)

                    model.fit(x_rnn_train, x_fc_train, y_train, x_rnn_valid,
                        x_fc_valid, y_valid,
                        epochs = epochs, batch_size = batch_size,
                        verbosity = 0, patience = 10)
                    model.print_summary()
                    model.plot_training()
                    model.save_model()

                    print('Model: %s,' % model.model_name,
                        ' with loss = %s,' % loss,
                        ' opt = %s,' % optimizer,
                        ' lr = %.4f' % lr)

                    eval = model.evaluate(x_rnn_test, x_fc_test, y_test, verbosity=2)

                    with open(path, 'a') as f:
                        f.write('Model: %s,' % model.model_name)
                        f.write(' with loss = %s,' % loss)
                        f.write(' opt = %s,' % optimizer)
                        f.write(' lr = %.4f: ' % lr)

                        f.write("%s: %.4f, " % (model.model.metrics_names[0], (eval[0])))
                        for i in range(1, len(eval)):
                            f.write("%s: %.2f%%, " % (model.model.metrics_names[i], (eval[i]*100)))
                        f.write('\n')

#Task 14:
def run_on_server_grid_search():
    start = time.process_time()
    path = '/cluster/scratch/cspreche/spotify_challenge'

    train_path = path + '/training_set_preproc'
    test_path = path + '/test_set_preproc'
    submission_path = path + '/submissions'

    tracks_path = train_path + '/log_8_20180902_000000000000.csv'
    sessions_path = train_path + '/session_log_8_20180902_000000000000.csv'

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    path_write = 'models/grid_search_' + now + '.txt'

    x_rnn, x_fc, y = utils.load_training_data_simple(tracks_path, sessions_path)

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

    print(x_rnn.shape)
    print(x_rnn_train.shape, x_rnn_valid.shape, x_rnn_test.shape)

    with open(path_write, 'a') as f:
        f.write('x_rnn: (%u,%u,%u)' % (x_rnn.shape[0],x_rnn.shape[1],x_rnn.shape[2]))
        f.write(', x_rnn_train: (%u)' % (x_rnn_train.shape[0]))
        f.write(', x_rnn_valid: (%u)' % (x_rnn_valid.shape[0]))
        f.write(', x_rnn_test: (%u) \n' % (x_rnn_test.shape[0]))

    del x_rnn, x_fc, y

    batch_size = 64
    epochs = 100

    grid_search(x_rnn_train, x_fc_train, y_train, x_rnn_valid,
            x_fc_valid, y_valid, x_rnn_test, x_fc_test, y_test,
            batch_size = batch_size, epochs = epochs, path = path_write)

    end = time.process_time()
    print("Time used: %4.2f seconds" % (end-start))

#Task 04:
def run_local_grid_search():
    start = time.process_time()

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    path_write = 'models/grid_search_' + now + '.txt'

    x_rnn, x_fc, y = utils.load_training_data_simple()

    print(x_rnn.shape)

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

    with open(path_write, 'a') as f:
        f.write('x_rnn: (%u,%u,%u)' % (x_rnn.shape[0],x_rnn.shape[1],x_rnn.shape[2]))
        f.write(', x_rnn_train: (%u)' % (x_rnn_train.shape[0]))
        f.write(', x_rnn_valid: (%u)' % (x_rnn_valid.shape[0]))
        f.write(', x_rnn_test: (%u) \n' % (x_rnn_test.shape[0]))

    del x_rnn, x_fc, y

    # Generate model
    batch_size = 64
    epochs = 25

    grid_search(x_rnn_train, x_fc_train, y_train, x_rnn_valid,
            x_fc_valid, y_valid, x_rnn_test, x_fc_test, y_test,
            batch_size = batch_size, epochs = epochs, path = path_write)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--task', required=True)

    io_args = parser.parse_args()
    task = io_args.task

    # Server commands
    if task == '10':
        run_on_server_simple()

    elif task == '11':
        continue_on_server_simple()

    elif task == '12':
        predict_on_server_simple()

    elif task == '13':
        run_on_server_using_generator()

    elif task == '14':
        run_on_server_grid_search()

    # Local commands
    elif task == '0':
        run_local_test()

    elif task == '3':
        run_local_using_generator()

    elif task == '4':
        run_local_grid_search()

    else:
        print('Choose Task')
