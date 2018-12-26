import numpy as np
import keras
from keras.models import Sequential, Model as K_Model, save_model, load_model
from keras.layers import (Input, Dense, Dropout, Flatten, Concatenate, Reshape,
    GRU, LSTM, concatenate, multiply)
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

import utils
from custom_losses_and_metrics import (selective_hinge as s_hinge,
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

    def fit(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

    def save_model(self):
        path = self.path + self.model_name + '_' + self.now + '.h5'
        save_model(self.model,path,overwrite=True,include_optimizer=True)

    def load_model(self, name):
        path = self.path + name + '.h5'
        self.model = load_model(path,
            custom_objects={'selective_hinge': s_hinge,
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
                plt.title('Model accuracy')
                plt.ylabel('Accuracy')
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


#hybrid models
class Hybrid(Model):

    def __init__(self, model_name = 'hybrid'):
        self.model_name = model_name
        self.set_shape()
        self.create_folder()
        self.set_timestamp()

    def build_model(self,
        rnn_layer_sizes = np.array([20, 20, 20]),
        dense_layer_parallel_sizes = np.array([10, 10]),
        dense_layer_sequential_sizes = np.array([32, 20]),
        dropout_prob_rnn = 0.3,
        dropout_prob_dense = 0.3,
        lambda_reg_dense = 0.001):

        self.rnn_layer_sizes = rnn_layer_sizes
        self.dense_layer_parallel_sizes = dense_layer_parallel_sizes
        self.dense_layer_sequential_sizes = dense_layer_sequential_sizes
        self.dropout_prob_rnn = dropout_prob_rnn
        self.dropout_prob_dense = dropout_prob_dense
        self.lambda_reg_dense = lambda_reg_dense

        if dense_layer_parallel_sizes[-1] != rnn_layer_sizes[-1]:
            print('Dimensions of last layers of RNN and of parallel dense network must agree!')
            return

        # define inputs
        tracks_input = Input(self.input_shape_tracks , dtype='float32', name='tracks_input')
        session_input = Input(self.input_shape_sessions , dtype='float32', name='session_input')

        # RNN side
        x_rnn = LSTM(self.rnn_layer_sizes[0],return_sequences=True)(tracks_input)
        for i in range(1, self.rnn_layer_sizes.size):
            x_rnn = LSTM(self.rnn_layer_sizes[i],return_sequences=True)(x_rnn)

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

        # merge RNN and dense side
        x = multiply([out_rnn, out_fc])

        for i in range(self.dense_layer_sequential_sizes.size - 1):
            x = Dense(self.dense_layer_sequential_sizes[i], activation='relu',
                kernel_regularizer=l2(self.lambda_reg_dense))(x)

        x = Dense(self.dense_layer_sequential_sizes[-1], activation='linear',
            kernel_regularizer=l2(self.lambda_reg_dense))(x)

        output = Reshape(self.output_shape, name = 'output')(x)

        # create model and compile it
        self.model = K_Model(inputs=[tracks_input, session_input], outputs=[output])
        self.model.compile(optimizer='Adam', loss=s_hinge,
            metrics=[ns_binary_acc, avg_mean_acc, fp_acc])

    def fit(self, x_train_rnn, x_train_fc, y_train, x_valid_rnn = None,
        x_valid_fc = None, y_valid = None, epochs=50, batch_size=64,
        patience = 5, verbosity = 0):

        # define callbacks
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
             ModelCheckpoint(filepath=self.path + self.model_name + '_{epoch:02d}_{val_loss:.4f}.h5',
                monitor='val_loss', save_best_only=False)]

        self.history = self.model.fit({'tracks_input': x_train_rnn, 'session_input': x_train_fc},
            {'output': y_train},
            validation_data=({'tracks_input': x_valid_rnn, 'session_input': x_valid_fc},
            {'output': y_valid}),
            epochs=epochs, batch_size=batch_size, callbacks = self.callbacks, verbose = verbosity)

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
