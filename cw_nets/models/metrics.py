from keras import backend as K


def precision(framework):
    """A wrapper for precision metric objects for deep learning frameworks.

    Arguments
    ---------
    framework : str
        Which deep learning framework you are using. Currently the only option
        is `"keras"`.

    Returns
    -------
    A metric function instantiated as required for use during model training.

    """
    if framework == "keras":
        return _keras_precision


def recall(framework):
    """A wrapper for recall metric objects for deep learning frameworks.

    Arguments
    ---------
    framework : str
        Which deep learning framework you are using. Currently the only option
        is `"keras"`.

    Returns
    -------
    A metric function instantiated as required for use during model training.

    """
    if framework == "keras":
        return _keras_recall


def f1_score(framework):
    """A wrapper for F1 score metric objects for deep learning frameworks.

    Arguments
    ---------
    framework : str
        Which deep learning framework you are using. Currently the only option
        is `"keras"`.

    Returns
    -------
    A metric function instantiated as required for use during model training.

    """
    if framework == "keras":
        return _keras_f1_score


def _keras_precision(y_true, y_pred):
    """Precision for foreground pixels.

    Calculates pixelwise precision TP/(TP + FP).

    """
    # count true positives
    truth = K.round(K.clip(y_true, K.epsilon(), 1))
    pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1))
    true_pos = K.sum(K.cast(K.all(K.stack([truth, pred_pos], axis=2), axis=2),
                            dtype='float32'))
    pred_pos_ct = K.sum(pred_pos) + K.epsilon()
    precision = true_pos/pred_pos_ct

    return precision


def _keras_recall(y_true, y_pred):
    """Precision for foreground pixels.

    Calculates pixelwise recall TP/(TP + FN).

    """
    # count true positives
    truth = K.round(K.clip(y_true, K.epsilon(), 1))
    pred_pos = K.round(K.clip(y_pred, K.epsilon(), 1))
    true_pos = K.sum(K.cast(K.all(K.stack([truth, pred_pos], axis=2), axis=2),
                            dtype='float32'))
    truth_ct = K.sum(K.round(K.clip(y_true, K.epsilon(), 1)))
    if truth_ct == 0:
        return 0
    recall = true_pos/truth_ct

    return recall


def _keras_f1_score(y_true, y_pred):
    """F1 score for foreground pixels ONLY.

    Calculates pixelwise F1 score for the foreground pixels (mask value == 1).
    Returns NaN if the model does not identify any foreground pixels in the
    image.

    """

    prec = _keras_precision(y_true, y_pred)
    rec = _keras_recall(y_true, y_pred)
    # Calculate f1_score
    f1_score = 2 * (prec * rec) / (prec + rec)

    return f1_score
