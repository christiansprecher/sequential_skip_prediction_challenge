import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Concatenate, GRU, LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.utils import plot_model
import matplotlib.pyplot as plt


# Model Hyperparameters
rnn_layer_sizes = np.array([20, 20, 20]);
dense_layer_parallel_sizes = np.array([10, 10]);
dense_layer_sequential_sizes = np.array([32, 20]);

dropout_prob_rnn = 0.3
dropout_prob_parallel_dense = 0.3
lambda_reg_dense = 0.001

input_shape_rnn = (20,1)
input_shape_dense = (10,1)

# Create Sequential RNN layers
model_sequential_rnn = Sequential()
model_sequential_rnn.add(LSTM(rnn_layer_sizes[0],return_sequences=True, input_shape=input_shape_rnn))
for i in range(1, rnn_layer_sizes.size):
    model_sequential_rnn.add(LSTM(rnn_layer_sizes[i], return_sequences=True))

model_sequential_rnn.add(BatchNormalization())
model_sequential_rnn.add(Dropout(dropout_prob_rnn))

# Create Sequential Dense layers
model_sequential_dense = Sequential()
model_sequential_dense.add(Dense(dense_layer_parallel_sizes[0],
    input_shape = input_shape_dense, activation='relu',kernel_regularizer=l2(lambda_reg_dense)))
for i in range(1, dense_layer_parallel_sizes.size):
    model_sequential_dense.add(Dense(dense_layer_parallel_sizes[i],
        activation='relu',kernel_regularizer=l2(lambda_reg_dense)))

model_sequential_dense.add(BatchNormalization())
model_sequential_dense.add(Dropout(dropout_prob_parallel_dense))

# Merge the two parallel models
model_full = Sequential()
model_full.add(Concatenate([model_sequential_rnn,model_sequential_dense]))
for i in range(dense_layer_sequential_sizes.size - 1):
    model_full.add(Dense(dense_layer_sequential_sizes[i],
        activation='relu',kernel_regularizer=l2(lambda_reg_dense)))
model_full.add(Dense(dense_layer_sequential_sizes[-1], activation='linear'))

# Selective Hinge Loss and Binary Accuracy
# y_pred should be -1 if not skipped, 0 if does not have to be predicted, 1 if skipped
def selective_hinge(y_true, y_pred):
    return keras.mean(keras.maximum(1. - y_true * y_pred, 0.) * keras.pow(y_true,2), axis=-1)
# Accuracy is also calculated for nodes not to be predicted (first half)
# Therefore, the best accuracy is about 0.5
def selective_binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.sign(y_pred)), axis=-1)

callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='model/keras_{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=False)]

model_full.compile(loss='selective_hinge', optimizer='Adam',metrics=['selective_binary_accuracy'])

plot_model(model_sequential_rnn, to_file='model/keras_rnn.png')
plot_model(model_sequential_dense, to_file='model/keras_dense.png')
plot_model(model_full, to_file='model/keras_full.png')
# print(model_full.summary())
