import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer

## Note
## TensorFlow's GPU SVD is too slow, so if you want to do end-to-end augmentation,
## I recommend to use cpu numpy (tensor version) augmentation.

class PCAColorAugmentationGPU(Layer):
    def __init__(self, std_deviation=0.1, clipping=True, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.std_deviation = std_deviation
        self.clipping = clipping
        self.scale = scale

    def build(self, input_shape):
        return super().build(input_shape)

    def augmentation(self, inputs):
        # assume channels-last
        input_shape = K.int_shape(inputs)
        ranks = len(input_shape)
        assert ranks >= 2 and ranks <=5
        chs = input_shape[-1]
        
        # swapaxis, reshape for calculating covariance matrix
        # rank 2 = (batch, dims)
        # rank 3 = (batch, step, dims)
        if ranks <= 3:
            x = inputs
        # rank 4 = (batch, height, width, ch) -> (batch, dims, ch)
        elif ranks == 4:
            dims = input_shape[1] * input_shape[2]
            x = tf.reshape(inputs, (-1, dims, chs))
        # rank 5 = (batch, D, H, W, ch) -> (batch, D, dims, ch)
        elif ranks == 5:
            dims = input_shape[2] * input_shape[3]
            depth = input_shape[1]
            x = tf.reshape(inputs, (-1, depth, dims, chs))

        # scaling-factor
        calculate_axis, reduce_axis = ranks-1, ranks-2
        if ranks == 3:
            calculate_axis, reduce_axis = 1, 2
        elif ranks >= 4:
            calculate_axis, reduce_axis = ranks-3, ranks-2
        C = 1.0
        if ranks >= 3:
            C = K.int_shape(x)[reduce_axis]
        var = K.var(x, axis=calculate_axis, keepdims=True)
        scaling_factors = tf.sqrt(C / tf.reduce_sum(var, axis=reduce_axis, keepdims=True))
        # scaling
        x = x * scaling_factors

        # subtract mean for cov matrix
        mean = tf.reduce_mean(x, axis=calculate_axis, keepdims=True)
        x -= mean

        # covariance matrix
        cov_n = max(K.int_shape(x)[calculate_axis] - 1, 1)
        # cov (since x was normalized)
        cov = tf.matmul(x, x, transpose_a=True) / cov_n

        # eigen value(S), eigen vector(U)
        S, U, V = tf.linalg.svd(cov)
        # eigen_values vs eigen_vectors
        # rank = 2 : (dims) vs (dims, dims)
        # rank = 3 : (batch, dims) vs (batch, dims, dims)
        # rank = 4 : (batch, ch) vs (batch, ch, ch)
        # rank = 5 : (batch, D, ch) vs (batch, D, ch, ch)
        
        # random values
        # if rank2 : get differnt random variable by sample
        if ranks == 2:
            rand = tf.random_normal(tf.shape(inputs), mean=0.0, stddev=self.std_deviation)
            delta = tf.matmul(rand*tf.expand_dims(S, axis=0), U)
        else:
            rand = tf.random_normal(tf.shape(S), mean=0.0, stddev=self.std_deviation)
            delta_original = tf.squeeze(tf.matmul(U, tf.expand_dims(rand*S, axis=-1)), axis=-1)

        # adjust delta shape
        # rank = 3 : (batch, dims) -> (batch, 1[step], dims)
        # rank = 4 : (batch, ch) -> (batch, 1[ndim], ch)
        # rank = 5 : (batch, D, ch) -> (batch, D, 1[ndim], ch)
        if ranks == 3:
            delta = tf.expand_dims(delta_original, axis=ranks-2)
        elif ranks >= 4:
            delta = K.expand_dims(delta_original, axis=ranks-3)
            # reshape to original input shape(if rank >= 4)
            # rank = 4 : (batch, ndim, ch) -> (batch, height, width, ch)
            # rank = 5 : (batch, D, ndim, ch) -> (batch, D, height, width, ch)
            delta = tf.broadcast_to(delta, tf.shape(x))
            delta = K.reshape(delta, [-1, *input_shape[1:]])
        
        # delta scaling
        delta = delta * self.scale

        #print("scaling factor", K.int_shape(scaling_factors))
        #print("mean", K.int_shape(mean))
        #print("cov", K.int_shape(cov))
        #print("S", K.int_shape(S))
        #print("U", K.int_shape(U))
        #print("rand", K.int_shape(rand))
        #print("delta_original", K.int_shape(delta_original))
        #print("delta", K.int_shape(delta))

        # clipping (if clipping=True)
        result = inputs + delta
        if self.clipping:
            result = tf.clip_by_value(result, 0.0, self.scale)

        return result

    def call(self, inputs):
        return K.in_train_phase(self.augmentation(inputs), inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

from keras.layers import Input
from keras.datasets import cifar10
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

def demo_keras_cifar():
    (_, _), (X, _) = cifar10.load_data()
    X = X / 255.0

    input = Input((32, 32, 3))
    x = PCAColorAugmentationGPU()(input)
    model = Model(input, x)

    # fix to train phase
    K.set_learning_phase(1)

    aug = model.predict(X[:100])
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    for i in range(100):
        ax = plt.subplot(10, 10, i+1)
        ax.imshow(aug[i])
        ax.axis("off")
    plt.show()


from sklearn.datasets import load_iris

def demo_keras_iris():
    data = load_iris()
    X = data["data"]

    input = Input((4,))
    x = PCAColorAugmentationGPU(clipping=False)(input)
    model = Model(input, x)

    X_aug = model.predict(X)
    indices = np.arange(X.shape[0])
    cm = plt.get_cmap("Set1")
    plt.figure(figsize=(8,8))
    for i in range(X.shape[1]):
        ax = plt.subplot(2, 2, i+1)
        ax.plot(indices, X_aug[:,i], ".", color=cm(i))
        ax.set_title(data["feature_names"][i])
    plt.show()

if __name__ == "__main__":
    demo_keras_cifar()
    #demo_keras_iris()
