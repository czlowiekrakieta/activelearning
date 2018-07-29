import tensorflow as tf
from keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [EarlyStopping(), ReduceLROnPlateau()]


def smallnet(filters=None, kernels=None, input_shape=(32, 32, 3), dense_final=1024):
    if filters is None:
        filters = [64, 64]
    if kernels is None:
        kernels = [5, 5]

    model = Sequential()
    for j, (f, k) in enumerate(zip(filters, kernels)):
        kwargs = {'filters': f, 'kernel_size': k, 'activation': 'relu'}
        if j == 0:
            kwargs['input_shape'] = input_shape
        model.add(Conv2D(**kwargs))
        model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(dense_final))
    model.add(Dense(10, activation='softmax'))
    model.compile('sgd', loss='sparse_categorical_crossentropy')
    return model