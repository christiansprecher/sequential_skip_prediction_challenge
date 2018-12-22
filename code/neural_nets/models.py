import numpy as np
import keras
from keras.models import Sequential, Model as K_Model
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate
from keras.layers import GRU, LSTM, concatenate, multiply
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

from custom_losses_and_metrics import (selective_hinge as s_hinge,
    selective_binary_accuracy as s_binary_acc,
    normed_selective_binary_accuracy as ns_binary_acc)

#class skeleton for models
class Model:

    def __init__(self):
        self.model_name = 'skeleton'
        self.set_input_shape()

    def set_input_shape(self):
        self.input_shape_tracks = (20,40,)
        self.input_shape_sessions = (10,)

    def build_model(self):
        pass

    def fit(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

    def save_model(self):
        path = 'models/' + self.model_name + '.h5'
        keras.models.save_model(self.model,path,overwrite=True,include_optimizer=True)

    def load_model(self):
        path = 'models/' + self.model_name + '.h5'
        self.model = keras.models.load_model(path)

    def plot_model(self):
        path = 'models/'+ self.model_name + '.png'
        keras.utils.plot_model(self.model, to_file=path)

    def print_summary(self):
        keras.utils.print_summary(self.model)

#hybrid models
class Hybrid(Model):

    def __init__(self, model_name = 'hybrid'):

        self.model_name = model_name
        self.set_input_shape()
        return

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

        output = Dense(self.dense_layer_sequential_sizes[-1], activation='linear',
            kernel_regularizer=l2(self.lambda_reg_dense))(x)

        # create model and compile it
        self.model = K_Model(inputs=[tracks_input, session_input], outputs=[output])
        self.model.compile(optimizer='Adam', loss=s_hinge,
            metrics=[s_binary_acc, ns_binary_acc])

        # define callbacks
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='model/keras_{epoch:02d}-{val_loss:.4f}.h5',
                monitor='val_loss', save_best_only=False)]

    def fit(self, x_train_rnn, x_train_fc, y_train, x_valid_rnn = None,
        x_valid_fc = None, y_valid = None, verbosity = 0):

        return self.model.fit({'tracks_input': x_train_rnn, 'session_input': x_train_fc},
            {'output': y_train},
            validation_data=({'tracks_input': x_valid_rnn, 'session_input': x_valid_fc},
            {'output': y_valid}),
            epochs=50, batch_size=32, callbacks = self.callbacks, verbose = verbosity)

    def evaluate(self, x_rnn, x_fc, y, verbosity=0):
        return self.model.evaluate({'tracks_input': x_rnn, 'session_input': x_fc},
            {'output': y}, verbose=verbosity)

    def predict(self, x_rnn, x_fc, verbosity=0):
        return self.model.predict({'tracks_input': x_rnn, 'session_input': x_fc},
            verbose=verbosity)
