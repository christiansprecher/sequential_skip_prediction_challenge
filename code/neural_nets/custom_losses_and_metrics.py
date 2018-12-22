import keras
from keras import backend as K


################### CUSTOM LOSSES ############################################

# Selective Hinge Loss and Binary Accuracy
# y_pred should be -1 if not skipped, 0 if does not have to be predicted, 1 if skipped
def selective_hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.) * K.pow(y_true,2), axis=-1)



################### CUSTOM METRICS ###########################################



# Accuracy is also calculated for nodes not to be predicted (first half)
# Therefore, the best accuracy is about 0.5
def selective_binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.sign(y_pred)), axis=-1)

# Exclude nodes not to predict
def normed_selective_binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.sign(y_pred)), axis=-1) / K.mean(K.abs(y_true),axis=-1)
