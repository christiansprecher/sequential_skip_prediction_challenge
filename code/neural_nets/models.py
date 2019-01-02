import numpy as np
import keras
from keras.models import Sequential, Model as K_Model, save_model, load_model
from keras.layers import (Input, Dense, Dropout, Flatten, Concatenate, Reshape,
    GRU, LSTM, RepeatVector, Permute, concatenate, multiply, maximum, add)
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import datetime
import glob
import random
from random import shuffle

import utils
from utils import Training_Data_Loader
from custom_losses_and_metrics import (selective_hinge as s_hinge,
    mean_hinge_accuracy as m_hinge_acc,
    logistic_loss as log_loss,
    mean_logistic_loss_accuracy as m_log_acc,
    selective_binary_accuracy as s_binary_acc,
    normed_selective_binary_accuracy as ns_binary_acc,
    average_mean_accuracy as avg_mean_acc,
    first_prediction_accuracy as fp_acc)

#class skeleton for models
class Model:

    def __init__(self, model_name = 'skeleton'):
        self.model_name = model_name
        self.set_shape()
        self.create_folder()
        self.set_timestamp()

    def set_shape(self):
        self.input_shape_tracks = (20,40,)
        self.input_shape_sessions = (10,)
        self.output_shape = (20,)

    def create_folder(self):
        self.path = 'models/' + self.model_name + '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def set_timestamp(self):
        now = datetime.datetime.now()
        self.now = now.strftime("%Y-%m-%d_%H:%M")

    def build_model(self):
        pass

    def generate_train_data(self, path, validation_nr = 0, batch_size = 64,
            iterations_per_file = 10):

        tracks_logs = sorted(glob.glob(path + "/training_set_preproc/log_*.csv"))

        del tracks_logs[validation_nr]
        shuffle(tracks_logs)

        n = len(tracks_logs)
        idx = 0

        while True:
            track = tracks_logs[idx]

            loader = Training_Data_Loader(track, batch_size)

            random_idx = random.sample(range(loader.n_slices), iterations_per_file)
            idx = (idx + 1 ) % n

            for k in range(len(random_idx)):

                x_rnn, x_fc, y = loader.get_data(random_idx[k])

                yield ({'tracks_input': x_rnn, 'session_input': x_fc},
                {'output': y})

    def generate_valid_data(self, path, validation_nr = 0, batch_size = 64):

        tracks_logs = sorted(glob.glob(path + "/training_set_preproc/log_*.csv"))

        track = tracks_logs[validation_nr]
        session = os.path.dirname(track) + '/session_' + os.path.basename(track)

        x_rnn, x_fc, y = utils.load_training_data_simple(track, session)

        n_slices = int(x_rnn.shape[0] / batch_size)
        random_idx = np.random.randint(x_rnn.shape[0], size=x_rnn.shape[0])

        while True:
            for k in range(n_slices):
                random_idx_slices = random_idx[k*batch_size:(k+1)*batch_size]

                batch_x_rnn = x_rnn[random_idx_slices,:,:]
                batch_x_fc = x_fc[random_idx_slices,:]
                batch_y = y[random_idx_slices,:]

                yield ({'tracks_input': batch_x_rnn, 'session_input': batch_x_fc},
                {'output': batch_y})

    def generate_test_data(self, path):

        tracks_logs = sorted(glob.glob(path + "/test_set_preproc/log_*.csv"))
        n = len(tracks_logs)
        idx = 0

        while True:
            track = tracks_logs[idx]
            session = os.path.dirname(track) + '/session_' + os.path.basename(track)

            x_rnn, x_fc = utils.load_test_data_simple(track, session)

            idx = (idx + 1 ) % n
            yield ({'tracks_input': x_rnn, 'session_input': x_fc})

    def compile(self, optimizer = 'Adam', loss = 's_hinge', lr = 0.001):
        optimizer_v = []
        if optimizer == 'Adam':
            optimizer_v = Adam(lr = lr)
        elif optimizer == 'SGD':
            optimizer_v = SGD(lr = lr)

        if loss == 's_hinge':
            self.model.compile(optimizer = optimizer_v, loss = s_hinge,
                metrics=[ns_binary_acc, avg_mean_acc, fp_acc])

        elif loss == 'm_hinge_acc':
            self.model.compile(optimizer = optimizer_v, loss = m_hinge_acc,
                metrics=[ns_binary_acc, avg_mean_acc, fp_acc])

        elif loss == 'log_loss':
            self.model.compile(optimizer = optimizer_v, loss = log_loss,
                metrics=[ns_binary_acc, avg_mean_acc, fp_acc])

        elif loss == 'm_log_acc':
            self.model.compile(optimizer = optimizer_v, loss = m_log_acc,
                metrics=[ns_binary_acc, avg_mean_acc, fp_acc])

        else:
            self.model.compile(optimizer = optimizer_v, loss = loss,
                metrics=[ns_binary_acc, avg_mean_acc, fp_acc])

    def fit(self, x_train_rnn, x_train_fc, y_train, x_valid_rnn = None,
            x_valid_fc = None, y_valid = None, epochs=50, batch_size=64,
            patience = 5, verbosity = 0):

        # define callbacks
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
             ModelCheckpoint(filepath=self.path + self.model_name + '_{epoch:02d}_{val_average_mean_accuracy:.4f}.h5',
                monitor='val_loss', save_best_only=False)]

        self.history = self.model.fit({'tracks_input': x_train_rnn, 'session_input': x_train_fc},
            {'output': y_train},
            validation_data=({'tracks_input': x_valid_rnn, 'session_input': x_valid_fc},
            {'output': y_valid}),
            epochs=epochs, batch_size=batch_size, callbacks = self.callbacks, verbose = verbosity)

        n_epochs = len(self.history.history['loss'])
        print('Model trained for %u epochs' % n_epochs)

        return self.history

    def fit_generator(self, path, epochs=50, batch_size=64, patience = 5,
        steps_per_epoch = 500, validation_steps = 100, verbosity = 0,
        iterations_per_file = 50):

        # define callbacks
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
             ModelCheckpoint(filepath=self.path + self.model_name + '_{epoch:02d}_{val_loss:.4f}.h5',
                monitor='val_loss', save_best_only=False)]

        n_files = len(glob.glob(path + "/training_set_preproc/log_*.csv"))
        n = random.randint(0,n_files - 1)

        self.history = self.model.fit_generator(
            self.generate_train_data(path, batch_size = batch_size,
            validation_nr = n, iterations_per_file = iterations_per_file),
            validation_data =
            self.generate_valid_data(path, batch_size = batch_size, validation_nr = n),
            epochs = epochs, callbacks = self.callbacks, verbose = verbosity,
            steps_per_epoch = steps_per_epoch, validation_steps = validation_steps)

        n_epochs = len(self.history.history['loss'])
        print('Model trained for %u epochs' % n_epochs)

        return self.history

    def evaluate(self, x_rnn, x_fc, y, verbosity=0):
        eval = self.model.evaluate({'tracks_input': x_rnn, 'session_input': x_fc},
            {'output': y}, verbose=verbosity)

        for i in range(1,len(eval)):
            print("%s: %.2f%%" % (self.model.metrics_names[i], (eval[i]*100)))

        return eval

    def predict(self, x_rnn, x_fc, verbosity=0, write_to_file = False,
        overwrite = True, path = None):
        y_pred =  self.model.predict({'tracks_input': x_rnn, 'session_input': x_fc},
            verbose=verbosity)

        if write_to_file:
            session_length = x_fc[:,0] * 10 + 10

            if path == None:
                path = '../../data/submissions'
            path = path + '/' + self.model_name + '_' + self.now + '.txt'
            utils.save_submission(y_pred,session_length, path, overwrite = overwrite)

    def save_model(self):
        path = self.path + self.model_name + '_' + self.now + '.h5'
        save_model(self.model,path,overwrite=True,include_optimizer=True)

    def load_model(self, name):
        path = self.path + name + '.h5'
        self.model = load_model(path,
            custom_objects={'selective_hinge': s_hinge,
            'mean_hinge_accuracy' : m_hinge_acc,
            'logistic_loss' : log_loss,
            'mean_logistic_loss_accuracy' : m_log_acc,
            'normed_selective_binary_accuracy' : ns_binary_acc,
            'average_mean_accuracy' : avg_mean_acc,
            'first_prediction_accuracy' : fp_acc})

    def plot_model(self):
        path = self.path + self.model_name + '_' + self.now + '_architecture.png'
        plot_model(self.model, to_file=path)

    def print_summary(self):
        print(self.model.summary())

    def plot_training(self):
        #Extract the metrics from the history
        keys = self.history.history.keys()
        for key in keys:
            if key.find('val_') > -1 and key.find('val_loss') == -1:
                # Plot training & validation accuracy values
                plt.plot(self.history.history[key[4:]])
                plt.plot(self.history.history[key])
                plt.title('Model ' + key[4:])
                plt.ylabel(key[4:])
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plot_name = (self.path + self.model_name + '_'
                    + self.now + '_' + key[4:] + '.png')
                plt.savefig(plot_name, bbox_inches='tight')
                plt.clf()


        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plot_name = (self.path + self.model_name + '_'
            + self.now + '_loss.png')
        plt.savefig(plot_name, bbox_inches='tight')


# hybrid models
class Hybrid(Model):

    def __init__(self, model_name = 'hybrid'):
        self.model_name = model_name
        self.set_shape()
        self.create_folder()
        self.set_timestamp()

    def build_model(self,
        rnn_layer_sizes = np.array([20, 20, 20]),
        dense_layer_parallel_sizes = np.array([10, 20]),
        dense_layer_sequential_sizes = np.array([32, 20]),
        dropout_prob_rnn = 0.3,
        dropout_prob_dense = 0.3,
        lambda_reg_rnn = 0.001,
        lambda_reg_dense = 0.001,
        merge = 'multiply'):

        self.rnn_layer_sizes = rnn_layer_sizes
        self.dense_layer_parallel_sizes = dense_layer_parallel_sizes
        self.dense_layer_sequential_sizes = dense_layer_sequential_sizes
        self.dropout_prob_rnn = dropout_prob_rnn
        self.dropout_prob_dense = dropout_prob_dense
        self.lambda_reg_rnn = lambda_reg_rnn
        self.lambda_reg_dense = lambda_reg_dense
        self.merge = merge

        if dense_layer_parallel_sizes[-1] != rnn_layer_sizes[-1]:
            print('Dimensions of last layers of RNN and of parallel dense network must agree!')
            return

        # define inputs
        tracks_input = Input(self.input_shape_tracks , dtype='float32', name='tracks_input')
        session_input = Input(self.input_shape_sessions , dtype='float32', name='session_input')

        # RNN side
        x_rnn = LSTM(self.rnn_layer_sizes[0],return_sequences=True,
            kernel_regularizer=l2(self.lambda_reg_rnn))(tracks_input)
        for i in range(1, self.rnn_layer_sizes.size):
            x_rnn = LSTM(self.rnn_layer_sizes[i],return_sequences=True,
            kernel_regularizer=l2(self.lambda_reg_rnn))(x_rnn)

        x_rnn = BatchNormalization()(x_rnn)
        out_rnn = Dropout(self.dropout_prob_rnn)(x_rnn)

        # dense side
        x_fc = Dense(self.dense_layer_parallel_sizes[0], activation='relu',
            kernel_regularizer=l2(self.lambda_reg_dense))(session_input)

        for i in range(1, self.dense_layer_parallel_sizes.size):
            x_fc = Dense(self.dense_layer_parallel_sizes[i], activation='relu',
                kernel_regularizer=l2(self.lambda_reg_dense))(x_fc)

        x_fc = BatchNormalization()(x_fc)
        out_fc = Dropout(self.dropout_prob_dense)(x_fc)

        x = []
        # merge RNN and dense side
        if self.merge == 'multiply':
            x = multiply([out_rnn, out_fc])
        elif self.merge == 'add':
            out_fc = RepeatVector(20)(out_fc)
            x = add([out_rnn, out_fc])
        elif self.merge == 'concatenate':
            out_fc = RepeatVector(20)(out_fc)
            x = concatenate([out_rnn, out_fc], axis = -1)
        elif self.merge == 'maximum':
            out_fc = RepeatVector(20)(out_fc)
            x = maximum([out_rnn, out_fc])
        else:
            print('Choose proper merge variation: multiply, add, concatenate or maximum')

        for i in range(self.dense_layer_sequential_sizes.size - 1):
            x = Dense(self.dense_layer_sequential_sizes[i], activation='relu',
                kernel_regularizer=l2(self.lambda_reg_dense))(x)

        x = Dense(self.dense_layer_sequential_sizes[-1], activation='linear',
            kernel_regularizer=l2(self.lambda_reg_dense))(x)

        output = Reshape(self.output_shape, name = 'output')(x)

        # create model and compile it
        self.model = K_Model(inputs=[tracks_input, session_input], outputs=[output])

class Single_RNN_Full(Model):

    def __init__(self, model_name = 'single_rnn_full'):
        self.model_name = model_name
        self.set_shape()
        self.create_folder()
        self.set_timestamp()

    def build_model(self,
        rnn_layer_sizes = np.array([20, 20, 20]),
        dense_layer_sequential_sizes = np.array([32, 20]),
        dropout_prob_rnn = 0.3,
        dropout_prob_dense = 0.3,
        lambda_reg_rnn = 0.001,
        lambda_reg_dense = 0.001):

        self.rnn_layer_sizes = rnn_layer_sizes
        self.dense_layer_sequential_sizes = dense_layer_sequential_sizes
        self.dropout_prob_rnn = dropout_prob_rnn
        self.dropout_prob_dense = dropout_prob_dense
        self.lambda_reg_rnn = lambda_reg_rnn
        self.lambda_reg_dense = lambda_reg_dense

        # define inputs
        tracks_input = Input(self.input_shape_tracks , dtype='float32', name='tracks_input')
        session_input = Input(self.input_shape_sessions , dtype='float32', name='session_input')

        # Concatenate sessions and tracks
        session_rep = RepeatVector(20)(session_input)
        x_input = concatenate([tracks_input, session_rep], axis = -1)

        # RNN part
        x = LSTM(self.rnn_layer_sizes[0],return_sequences=True,
            kernel_regularizer=l2(self.lambda_reg_rnn))(x_input)
        for i in range(1, self.rnn_layer_sizes.size):
            x = LSTM(self.rnn_layer_sizes[i],return_sequences=True,
            kernel_regularizer=l2(self.lambda_reg_rnn))(x)

        x = BatchNormalization()(x)
        x = Dropout(self.dropout_prob_rnn)(x)

        for i in range(self.dense_layer_sequential_sizes.size - 1):
            x = Dense(self.dense_layer_sequential_sizes[i], activation='relu',
                kernel_regularizer=l2(self.lambda_reg_dense))(x)

        x = Dense(self.dense_layer_sequential_sizes[-1], activation='linear',
            kernel_regularizer=l2(self.lambda_reg_dense))(x)

        output = Reshape(self.output_shape, name = 'output')(x)

        # create model
        self.model = K_Model(inputs=[tracks_input, session_input], outputs=[output])
