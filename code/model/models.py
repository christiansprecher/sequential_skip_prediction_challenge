import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Concatenate, GRU, LSTM
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.utils import plot_model

import custom_losses_and_metrics

#class skeleton for models
class Model:

    def __init__(self):
        self.model_name = 'skeleton'
        pass

    def build_model(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def save_model(self):
        path = 'backup/' + self.model_name + '.h5'
        keras.models.save_model(self.model,path,overwrite=True,include_optimizer=True)

    def load_model(self):
        path = 'backup/' + self.model_name + '.h5'
        self.model = keras.models.load_model(path)



class Hybrid(Model):

    def __init__(self,
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

        self.model_name = 'hybrid'
        return

    def build_model(self):
        # Create Sequential RNN layers
        model_sequential_rnn = Sequential()
        model_sequential_rnn.add(LSTM(self.rnn_layer_sizes[0],return_sequences=True, input_shape=input_shape_rnn))
        for i in range(1, self.rnn_layer_sizes.size):
            model_sequential_rnn.add(LSTM(self.rnn_layer_sizes[i], return_sequences=True))

        model_sequential_rnn.add(BatchNormalization())
        model_sequential_rnn.add(Dropout(self.dropout_prob_rnn))

        # Create Sequential Dense layers
        model_sequential_dense = Sequential()
        model_sequential_dense.add(Dense(self.dense_layer_parallel_sizes[0],
            input_shape = input_shape_dense, activation='relu',
            kernel_regularizer=l2(self.lambda_reg_dense)))
        for i in range(1, self.dense_layer_parallel_sizes.size):
            model_sequential_dense.add(Dense(self.dense_layer_parallel_sizes[i],
                activation='relu',kernel_regularizer=l2(self.lambda_reg_dense)))

        model_sequential_dense.add(BatchNormalization())
        model_sequential_dense.add(Dropout(self.dropout_prob_parallel_dense))

        # Merge the two parallel models
        model_full = Sequential()
        model_full.add(Concatenate([model_sequential_rnn,model_sequential_dense]))
        for i in range(self.dense_layer_sequential_sizes.size - 1):
            model_full.add(Dense(self.dense_layer_sequential_sizes[i],
                activation='relu',kernel_regularizer=l2(self.lambda_reg_dense)))
        model_full.add(Dense(self.dense_layer_sequential_sizes[-1], activation='linear'))

        self.model = model_full

        self.callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='model/keras_{epoch:02d}-{val_loss:.4f}.h5',
                monitor='val_loss', save_best_only=False)]

        return


    def fit(self, x_train_rnn, x_train_fc, y_train, x_test_rnn = [],
        x_test_fc = [], y_test = []):

        pass
