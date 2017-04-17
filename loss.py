from keras import backend as K


def huber_loss(y_true, y_pred):
    """
    Using huber loss because it is less sensitive to outliers than mean squared error.
    It acts as mean squared error between -1 and 1 and behaves like mean absolute error
    when it is less than -1 and greater than 1.
    """
    return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)
