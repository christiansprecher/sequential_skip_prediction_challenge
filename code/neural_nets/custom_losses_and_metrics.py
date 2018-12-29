import keras
from keras import backend as K
import numpy as np


################### CUSTOM LOSSES ############################################

# Selective Hinge Loss and Binary Accuracy
# y_pred should be -1 if not skipped, 0 if does not have to be predicted, 1 if skipped
def selective_hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.) * K.abs(y_true,2), axis=-1)

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


############### NUMPY CUSTOM METRICS ###########################################
# Calculate mean average accuracy
def np_cumulative_binary_accuracy(y_true, y_pred):

    y_score = np.multiply(y_true, np.sign(y_pred)).astype(int)
    y_score_01 = np.array(y_score > 0).astype(int)

    a = ((y_true!=0).argmax(axis=1)).astype(int)
    b = (y_true.shape[1] - ((np.flip(y_true,1)!=0).argmax(axis=1))).astype(int)

    y_accuracy = np.zeros([y_true.shape[0],y_true.shape[1]])
    scores = np.zeros(y_true.shape[0])

    for i in range(y_true.shape[0]):
        for j in range(a[i],b[i]):
            print(i, j, np.sum(y_score_01[i,a[i]:(j+1)]))
            y_accuracy[i,j] = (np.sum(y_score_01[i,a[i]:(j+1)]) * y_score_01[i,j] / (j - a[i] + 1))
        scores[i] = np.sum(y_accuracy[i,:]) / (b[i]-a[i])

    return np.mean(scores)

def np_cumulative_binary_accuracy2(y_true, y_pred):

    y_score = np.multiply(y_true, np.sign(y_pred)).astype(int)
    y_score_01 = np.array(y_score > 0).astype(int)
    y_score_abs = np.abs(y_score)

    a = np.array((y_true!=0).argmax(axis=1)).astype(int)
    a = np.expand_dims(a, axis=1)
    y_a = np.tile(a,(1,y_true.shape[1]))

    cumsum = np.cumsum(np.ones([y_true.shape[0],y_true.shape[1]]),axis=1)

    y_cumsum = np.cumsum(y_score_01,axis=1)

    y_weight = cumsum - y_a
    #avoid zeros in weight
    y_weight = y_weight + (np.ones([y_true.shape[0],y_true.shape[1]]) - y_score_abs) * y_true.shape[1]

    y_accuracy = np.multiply(y_cumsum,y_score_01)

    y_normed = np.divide(y_accuracy,y_weight)

    y_numbers = np.tile(np.expand_dims(np.sum(y_score_abs,axis=1),axis=1),(1,y_true.shape[1]))
    y_more_normed = np.divide(y_normed,y_numbers)

    scores = np.sum(y_more_normed,axis=1)

    return scores
