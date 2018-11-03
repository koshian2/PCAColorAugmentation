from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, AveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import pickle

from pca_aug_tf_keras_version import PCAColorAugmentationGPU

def create_basic_block_simple(input, chs, use_batchnorm=True):
    x = Conv2D(chs, kernel_size=3, padding="same")(input)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def create_regularized_block(input, chs, **kwargs):
    x = Conv2D(chs, kernel_size=3, padding="same")(input)
    x = PCAColorAugmentationGPU(clipping=False)(x)
    x = Activation("relu")(x)
    return x

def create_model(mode, use_batchnorm=True):
    assert mode in ["no", "augment", "full"]

    if mode in ["no", "augment"]:
        block_func = create_basic_block_simple
    elif mode == "full":
        block_func = create_regularized_block

    input = Input(shape=(32, 32, 3))
    if mode != "no":
        x = PCAColorAugmentationGPU(clipping=True)(input)
    else:
        x = input
    x = block_func(x, 32, use_batchnorm=use_batchnorm)
    x = AveragePooling2D(2)(x)
    x = block_func(x, 64, use_batchnorm=use_batchnorm)
    x = AveragePooling2D(2)(x)
    x = block_func(x, 128, use_batchnorm=use_batchnorm)
    x = Flatten()(x)
    x = Dense(10, activation="softmax")(x)
    return Model(input, x)

def train(mode, use_batchnorm=True):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = X_train/255.0, X_test/255.0
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    model = create_model(mode, use_batchnorm)
    model.summary()
    model.compile(SGD(0.01, 0.9), loss="categorical_crossentropy", metrics=["acc"])

    history = model.fit(X_train, y_train, batch_size=512, validation_data=(X_test, y_test), epochs=100)

    bn_flag = "" if use_batchnorm else "_no_bn"
    with open(f"history_{mode}{bn_flag}.dat", "wb") as fp:
        pickle.dump(history.history, fp)

if __name__ == "__main__":
    train("augment", False)

    # Seconds per epoch (GPU on Colab)
    # PCA : No, BatchNorm : No -> 6s
    # PCA : No, BatchNorm : Yes -> 7s
    # PCA : Yes, BatchNorm : No -> 92s
    # PCA : Yes, BatchNorm : Yes -> 94s

    # full mode is extremely slow... (no = 7s/epoch vs full = over 2hrs)
    #  (due to SVD on GPU is much slow than on CPU)
    # https://github.com/tensorflow/tensorflow/issues/13603

    # It may not be realistic to train through a PCA augmentation layer with BackProp.

    # If it happens, you should try CPU augmentation (cnn_cifar.py).
