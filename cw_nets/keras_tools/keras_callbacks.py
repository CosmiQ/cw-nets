## Code modified from https://github.com/CosmiQ/basiss/blob/master/src/basiss.py
from keras import backend as K

###############################################################################
### Jaccard
###############################################################################
def jaccard_coef(y_true, y_pred, smooth=1e-12):
    '''https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
    # __author__ = Vladimir Iglovikov'''
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

###############################################################################
def jaccard_coef_int(y_true, y_pred, smooth=1e-12):
    '''https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
    # __author__ = Vladimir Iglovikov'''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

###############################################################################
###############################################################################
### Manually define metrics
# define metrics from https://github.com/fchollet/keras/blob/master/keras/metrics.py
#   since for some reason keras can't always import them
###############################################################################    
def dice_coeff(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersect = K.sum(y_true_flat * y_pred_flat)
    return (2. * intersect) / (K.sum(y_true_flat) + K.sum(y_pred_flat))

###############################################################################
def dice_loss(y_true, y_pred):
    return -1. * dice_coeff(y_true, y_pred)
    
###############################################################################
def mse(y_true, y_pred):
    return K.mean(K.square(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)

###############################################################################
def f1_score(y_true, y_pred):
    '''https://stackoverflow.com/questions/45411902/how-to-use-f1-score-with-keras-model'''

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score 

###############################################################################
def f1_loss(y_true, y_pred):
    return 1. - f1_score(y_true, y_pred)

