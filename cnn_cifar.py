from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, AveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle

from pca_aug_numpy_tensor import pca_color_augmentation_tensor

class PCAAugmentGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def flow(self, x, y, enable_pca_augmentation, **kwargs):
        for X_batch, y_batch in super().flow(x, y=y, **kwargs):
            X_batch = pca_color_augmentation_tensor(X_batch)
            yield X_batch, y_batch

def create_basic_block_simple(input, chs, use_batchnorm=True):
    x = Conv2D(chs, kernel_size=3, padding="same")(input)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def create_model(use_batchnorm):
    input = Input(shape=(32, 32, 3))
    x = create_basic_block_simple(input, 32, use_batchnorm=use_batchnorm)
    x = AveragePooling2D(2)(x)
    x = create_basic_block_simple(x, 64, use_batchnorm=use_batchnorm)
    x = AveragePooling2D(2)(x)
    x = create_basic_block_simple(x, 128, use_batchnorm=use_batchnorm)
    x = Flatten()(x)
    x = Dense(10, activation="softmax")(x)
    return Model(input, x)

def train(mode, use_batchnorm=True):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    model = create_model(use_batchnorm)
    model.summary()
    model.compile(SGD(0.01, 0.9), loss="categorical_crossentropy", metrics=["acc"])

    gen = PCAAugmentGenerator(rescale=1.0/255)

    history = model.fit_generator(gen.flow(X_train, y_train, True, batch_size=512),  steps_per_epoch=50000/512,
                                  validation_data=gen.flow(X_test, y_test, False, batch_size=512), validation_steps=10000//512,
                                  epochs=100)

    bn_flag = "" if use_batchnorm else "_no_bn"
    with open(f"history_{mode}{bn_flag}.dat", "wb") as fp:
        pickle.dump(history.history, fp)

if __name__ == "__main__":
    train("augment", False)

    # Seconds per epoch (Train on Colab GPU and augmented on CPU)
    # PCA : Yes, BatchNorm : No -> 14s 
    # PCA : Yes, BatchNorm : Yes -> 14s

    # Not bad!
    # CPU augmentation is about x7 faster than GPU augmentation!!
