import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import enum

class TripletTechnique(enum):
    AUGMENTATION = 1
    IMAGE_AUGMENTATION = 2


class AbstractIterationTechnique(object):
    pass


class SequentialIterationTechnique(AbstractIterationTechnique):
    def __init__(self):
        self.index = 0

    def increment(self, n=1):
        self.index += n

    def reset(self):
        self.index = 0


class RandomIterationTechnique(AbstractIterationTechnique):
    pass


class TripletDataset(object):
    def __init__(self, r, p, n, r_class, p_class, n_class, weights):
        self.r = r
        self.p = p
        self.n = n
        self.r_class = r_class
        self.p_class = p_class
        self.n_class = n_class
        self.weights = weights

    def get_reference(self):
        return None

    def get_positive(self):
        return None

    def get_negative(self):
        return None

    def get_reference_class(self):
        return None

    def get_positive_class(self):
        return None

    def get_negative_class(self):
        return None

    def get_weights(self):
        return None

class AbstractGenerator(object):
    def __next_image(self):
        raise BaseException("not implemented")


class RotatedMNISTDataGenerator(AbstractGenerator):
    def __init__(self,
                 train_ratio=0.85,
                 valid_ratio=0.05,
                 test_ratio=0.1,
                 triplet_technique=TripletTechnique.AUGMENTATION,
                 train_iteration_technique=RandomIterationTechnique):
        self.train_images = []
        self.valid_images = []
        self.test_images = []
        self.triplet_technique = triplet_technique
        self.train_images = train_iteration_technique()
        if train_ratio + valid_ratio + test_ratio > 1.0:
            raise BaseException('cannot have ratios go greater than 1.0')

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        images = []
        for image, label in zip(mnist.test.images, mnist.test.labels):
            images.append([label, np.reshape(image, (28, 28, 1))])
        images = np.array(images)

        valid_ratio += train_ratio
        test_ratio += valid_ratio

        train_idx = int(len(images) * train_ratio)
        valid_idx = int(len(images) * valid_ratio)

        self.train_images = images[:train_idx]
        self.valid_images = images[train_idx:valid_idx]
        self.test_images = images[valid_idx:test_ratio]

    def train(self, batch_size):
        return None, None

    def triplet_train(self, batch_size):
        return None

    def test(self, batch_size=None):
        return None, None

    def validation(self, batch_size=None):
        return None

    def reset(self):
        pass

    def data_shape(self):
        return (28, 28)