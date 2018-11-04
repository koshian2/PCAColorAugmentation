# PCA Color Augmentation
PCA Color Augmentation described in [the AlexNet's paper](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf). You can run with TensorFlow, Keras, Numpy.

# What's PCA Color Augmentation
It is a kind of Data Augmentation and uses principal component analysis. By calculating eigenvectors and eigenvalues, it is possible to add noise matching the distribution of data, and behave like if there are many images (data).

![](https://github.com/koshian2/PCAColorAugmentation/blob/master/images/cifar.gif)

# For structured data
The original PCA Augmentation is only an image, I notice that this can be applied to structured data, so I implement it.

## Simple Augmentation
![](https://github.com/koshian2/PCAColorAugmentation/blob/master/images/iris.gif)

## Categorical Augmentation
Since it is implemented as a tensor calculation, it is possible to augment structured data, not just images, by category

![](https://github.com/koshian2/PCAColorAugmentation/blob/master/images/iris.gif)

## Evaluation
### CIFAR-10
AlexNet is pre-BatchNorm paper, so I check the existence of BatchNorm besides PCA Augmentation.

| PCA　Augmenation | Batch Norm | Train Acc | Validation Acc | s/epc (GPU) | s/epc (CPU) |
|:----------------:|:----------:|----------:|---------------:|:-----------:|:-----------:|
|        Yes       |     Yes    |    0.9931 |         0.7618 |      94     |      14     |
|        Yes       |     No     |    0.9208 |         0.6651 |      92     |      14     |
|        No        |     Yes    |    1.0000 |         0.7762 |      6      |      -      |
|        No        |     No     |    0.9843 |         0.6507 |      6      |      -      |

BatchNorm is too strong. The effect of augmentation is canceled by BatchNorm on this evaluation.
Without BatchNorm, it is certainly possible to confirm the effect of PCA Augmentation.

s/epc (GPU) means seconds per epoch when PCA Augmentation is run on GPU. s/epc (CPU) is same on CPU. **Both train on GPU**.

The reason why the GPU version is slow is because the SVD of the TensorFlow GPU is very slow. [Related issue](https://github.com/tensorflow/tensorflow/issues/13603).

So, **I recommend running PCA Augmentation with Numpy tensor version of CPU (pca_aug_numpy_tensor.py**).

### Wine Dataset(For structured data)
[Scikit-learn wine datasets](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine). To reproduce overfitting, I change the train and test split rate from general 7: 3 to 3: 7.

| PCA Augmentation | # PCA | MaxValAcc |
|------------------|------:|----------:|
| No               |     - |    0.9440 |
| Yes(Total)       |     5 |    0.9600 |
| Yes(Total)       |    20 |    0.9760 |
| Yes(Categorical) |     5 |    0.9600 |
| Yes(Categorical) |    20 |    0.9680 |

**PCA Augmentation for structured data does well!** Categorical augmentation is works well too.

## See details (Japanese)
* [https://blog.shikoan.com/pca-color-augmentation/](https://blog.shikoan.com/pca-color-augmentation/)
* [https://qiita.com/koshian2/items/78de8ccd09dd2998ddfc](https://qiita.com/koshian2/items/78de8ccd09dd2998ddfc)
