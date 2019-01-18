import keras
from keras import backend as K
import numpy as np


################### CUSTOM LOSSES ############################################

# Selective Hinge Loss
# y_pred should be -1 if not skipped, 0 if does not have to be predicted, 1 if skipped
def selective_hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.) * K.abs(y_true), axis=-1)

# Weighted Hinge
# Use selective hinge, but apply with same weight as average mean accuracy
def mean_hinge_accuracy(y_true, y_pred):
    ones = K.ones_like(y_true,'float32')
    # dim = K.int_shape(ones)[1]
    dim = 20

    y_hinge = K.maximum(1. - y_true * y_pred, 0.)
    y_score_pos = K.abs(y_true)
    y_score_neg = ones - K.abs(y_true)

    num_predictions = K.expand_dims(K.sum(y_score_pos, axis = 1), axis = 1)
    num_predictions_rep = K.repeat_elements(num_predictions, rep = dim, axis = 1)

    predict_start = K.cast( K.expand_dims(K.argmax(y_score_pos, axis = 1),
        axis = 1), 'float32')
    predict_start_rep = K.repeat_elements(predict_start, rep = dim, axis = 1)

    cumsum = K.cumsum(ones,axis=1)
    weights = cumsum - predict_start_rep + y_score_neg * dim

    y_cumsum = K.cumsum(y_hinge,axis=1)
    y_clear = y_cumsum * y_score_pos
    y_normed = (y_clear / weights) / num_predictions_rep
    return K.sum(y_normed,axis=1)

# Logistic loss which goes over all elements
def logistic_loss(y_true, y_pred):
    return K.mean(K.log(K.ones_like(y_true,'float32') + K.exp(- y_true * y_pred)), axis = -1)

# Logistic loss weighted similar to accuracy
def mean_logistic_loss_accuracy(y_true, y_pred):
    ones = K.ones_like(y_true,'float32')
    dim = 20

    y_log_loss = K.log(ones + K.exp(- y_true * y_pred))
    y_score_pos = K.abs(y_true)
    y_score_neg = ones - K.abs(y_true)

    num_predictions = K.expand_dims(K.sum(y_score_pos, axis = 1), axis = 1)
    num_predictions_rep = K.repeat_elements(num_predictions, rep = dim, axis = 1)

    predict_start = K.cast( K.expand_dims(K.argmax(y_score_pos, axis = 1),
        axis = 1), 'float32')
    predict_start_rep = K.repeat_elements(predict_start, rep = dim, axis = 1)

    cumsum = K.cumsum(ones,axis=1)
    weights = cumsum - predict_start_rep + y_score_neg * dim

    y_cumsum = K.cumsum(y_log_loss,axis=1)
    y_clear = y_cumsum * y_score_pos
    y_normed = (y_clear / weights) / num_predictions_rep
    return K.sum(y_normed,axis=1)


################### CUSTOM METRICS ###########################################


# Accuracy is also calculated for nodes not to be predicted (first half)
# Therefore, the best accuracy is about 0.5
def selective_binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.sign(y_pred)), axis=-1)

# Exclude nodes not to predict
def normed_selective_binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.sign(y_pred)), axis=-1) / K.mean(K.abs(y_true),axis=-1)

# Use average mean accuracy
def average_mean_accuracy(y_true, y_pred):
    ones = K.ones_like(y_true,'float32')
    # dim = K.int_shape(ones)[1]
    dim = 20

    y_score = y_true * K.sign(y_pred)
    y_score_pos = K.abs(y_score)
    y_score_neg = ones - K.abs(y_score)
    y_score_01 = K.cast(K.greater_equal(y_score, ones),'float32')

    num_predictions = K.expand_dims(K.sum(y_score_pos, axis = 1), axis = 1)
    num_predictions_rep = K.repeat_elements(num_predictions, rep = dim, axis = 1)

    predict_start = K.cast( K.expand_dims(K.argmax(y_score_pos, axis = 1),
        axis = 1), 'float32')
    predict_start_rep = K.repeat_elements(predict_start, rep = dim, axis = 1)

    cumsum = K.cumsum(ones,axis=1)
    weights = cumsum - predict_start_rep + y_score_neg * dim

    y_cumsum = K.cumsum(y_score_01,axis=1)
    y_accuracy = y_cumsum * y_score_01
    y_normed = (y_accuracy / weights) / num_predictions_rep
    return K.sum(y_normed,axis=1)

#Accuracy on first prediction of second half of session
def first_prediction_accuracy(y_true, y_pred):
    ones = K.ones_like(y_true,'float32')
    # dim = K.int_shape(ones)[1]
    dim = 20

    y_score = y_true * K.sign(y_pred)
    y_score_pos = K.abs(y_score)
    y_score_01 = K.cast(K.greater_equal(y_score, ones),'float32')

    predict_start = K.cast( K.expand_dims(K.argmax(y_score_pos, axis = 1),
        axis = 1), 'float32')
    predict_start_rep = K.repeat_elements(predict_start, rep = dim, axis = 1)

    cumsum = K.cumsum(ones,axis=1)
    first_prediction = K.cast(K.equal(predict_start_rep, cumsum - ones),'float32')
    first_prediction_acc = y_score_01 * first_prediction

    return K.sum(first_prediction_acc,axis=1)
