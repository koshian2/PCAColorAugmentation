import numpy as np

def pca_color_augmentation_numpy(image_array_input):
    assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3
    assert image_array_input.dtype == np.uint8

    img = image_array_input.reshape(-1, 3).astype(np.float32)
    scaling_factor = np.sqrt(3.0 / np.sum(np.var(img, axis=0)))
    img *= scaling_factor

    cov = np.cov(img, rowvar=False)
    U, S, V = np.linalg.svd(cov)

    rand = np.random.randn(3) * 0.1
    delta = np.dot(U, rand*S)
    delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)
    return img_out

from keras.datasets import cifar10
import matplotlib.pyplot as plt

def demo_numpy(X):
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    for i in range(100):
        ax = plt.subplot(10, 10, i+1)
        aug = pca_color_augmentation_numpy(X[i])
        ax.imshow(aug)
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    (_, _), (X, _) = cifar10.load_data()
    for i in range(10):
        demo_numpy(X)
