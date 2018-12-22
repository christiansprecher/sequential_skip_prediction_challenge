import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate
from keras.layers import GRU, LSTM, concatenate, multiply
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.utils import plot_model

import custom_losses_and_metrics

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
        plot_model(self.model, to_file=path)

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
        lambda_reg_dense = 0.001,):

        self.rnn_layer_sizes = rnn_layer_sizes
        self.dense_layer_parallel_sizes = dense_layer_parallel_sizes
        self.dense_layer_sequential_sizes = dense_layer_sequential_sizes
        self.dropout_prob_rnn = dropout_prob_rnn
        self.dropout_prob_dense = dropout_prob_dense
        self.lambda_reg_dense = lambda_reg_dense

        # define inputs
        tracks_input = Input(self.input_shape_tracks , dtype='float32', name='tracks_input')
        session_input = Input(self.input_shape_sessions , dtype='float32', name='session_input')

        # RNN side
        x_rnn = LSTM(self.rnn_layer_sizes[0],return_sequences=True)(tracks_input)
        for i in range(1, self.rnn_layer_sizes.size):
            x_rnn = LSTM(self.rnn_layer_sizes[i],return_sequences=True)(x_rnn)

        x_rnn = BatchNormalization()(x_rnn)
        out_rnn = Dropout(dropout_prob_rnn)(x_rnn)

        # dense side
        x_fc = Dense(self.dense_layer_parallel_sizes[0], activation='relu',
            kernel_regularizer=l2(lambda_reg_dense))(session_input)

        for i in range(1, dense_layer_parallel_sizes.size):
            x_fc = Dense(dense_layer_parallel_sizes[i], activation='relu',
                kernel_regularizer=l2(lambda_reg_dense))(x_fc)

        x_fc = BatchNormalization()(x_fc)
        out_fc = Dropout(dropout_prob_parallel_dense)(x_fc)

        # merge RNN and Dense side
        x = multiply([out_rnn, out_fc])

        for i in range(dense_layer_sequential_sizes.size - 1):
            x = Dense(dense_layer_parallel_sizes[i], activation='relu',
                kernel_regularizer=l2(lambda_reg_dense))(x)

        output = Dense(dense_layer_parallel_sizes[-1], activation='linear',
            kernel_regularizer=l2(lambda_reg_dense))(x)

        # create model and compile it
        self.model = Model(inputs=[tracks_input, session_input], outputs=[output])
        self.model.compile(optimizer='Adam', loss=selective_hinge,
            metrics=[selective_binary_accuracy,normed_selective_binary_accuracy])

        # define callbacks
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='model/keras_{epoch:02d}-{val_loss:.4f}.h5',
                monitor='val_loss', save_best_only=False)]

    def fit(self, x_train_rnn, x_train_fc, y_train, x_test_rnn = None,
        x_test_fc = None, y_test = None, verbosity=0):

        self.model.fit({'tracks_input': x_train_rnn, 'sessions_input': x_train_fc},
            {'output': y_train},
            validation_data={'tracks_input': x_test_rnn, 'sessions_input': x_test_fc},
            {'output': y_test},
            epochs=50, batch_size=32, callbacks = self.callbacks, verbose = verbosity)

    def evaluate(self, x_rnn, x_fc, y, verbosity=0):
        return self.model.evaluate({'tracks_input': x_rnn, 'sessions_input': x_fc},
            {'output': y}, verbose=verbosity)

    def predict(self, x_rnn, x_fc, verbosity=0):
        return self.model.predict({'tracks_input': x_rnn, 'sessions_input': x_fc},
            verbose=verbosity)
