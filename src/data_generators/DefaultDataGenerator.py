import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from enum import Enum
import random

class TripletTechnique(Enum):
    AUGMENTATION = 1
    IMAGE_AUGMENTATION = 2


class AbstractIterationTechnique(object):
    pass


class SequentialIterationTechnique(AbstractIterationTechnique):
    def __init__(self):
        self.used = None
        self.mask = None

    def reset(self, dataset):
        self.mask = np.array(map(lambda _: True, dataset))



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
        return self.r

    def get_positive(self):
        return self.p

    def get_negative(self):
        return self.p

    def get_reference_class(self):
        return self.r_class

    def get_positive_class(self):
        return self.p_class

    def get_negative_class(self):
        return self.n_class

    def get_weights(self):
        return self.weights

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
        self.train_image_classes = []
        self.valid_image_classes = []
        self.test_image_classes = []

        self.triplet_technique = triplet_technique
        self.train_images_iteration_technique = train_iteration_technique()
        if train_ratio + valid_ratio + test_ratio > 1.0:
            raise BaseException('cannot have ratios go greater than 1.0')

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        images = []
        labels = []
        for image, label in zip(mnist.test.images, mnist.test.labels):
            images.append(np.reshape(image, (28, 28, 1)))
            labels.append(label)
        images = np.array(images)
        labels = np.array(labels)

        valid_ratio += train_ratio
        test_ratio += valid_ratio

        train_idx = int(len(images) * train_ratio)
        valid_idx = int(len(images) * valid_ratio)
        test_idx = int(len(images) * test_ratio)

        self.train_images = images[:train_idx]
        self.train_image_classes = labels[:train_idx]

        self.valid_images = images[train_idx:valid_idx]
        self.valid_image_classes = labels[train_idx:valid_idx]

        self.test_images = images[valid_idx:test_idx]
        self.test_image_classes = labels[valid_idx:test_idx]

        if type(self.train_images_iteration_technique) is SequentialIterationTechnique:
            self.train_images_iteration_technique.reset(self.train_images)

    def train(self, batch_size):
        if type(self.train_images_iteration_technique) is RandomIterationTechnique:
            indices = np.random.choice(self.train_images.shape[0], size=batch_size)
            return self.train_images[indices], self.train_image_classes[indices]

        elif type(self.train_images_iteration_technique) is SequentialIterationTechnique:
            mask = self.train_images_iteration_technique.mask
            size = len(mask)
            idxs = np.array(range(0, size))[mask]
            if len(idxs) == 0:
                return None
            idxs = idxs[:max(len(idxs), batch_size)]
            mask[idxs] = False
            return self.train_images[idxs], self.train_image_classes[idxs]

        else:
            raise BaseException('unknown iteration technique')

    def triplet_train(self, batch_size):
        if self.triplet_technique == TripletTechnique.AUGMENTATION:
            reference =
        elif self.triplet_technique == TripletTechnique.IMAGE_AUGMENTATION:
            pass

    def validation(self, batch_size=None):
        idx = len(self.test_images) if batch_size is None else batch_size
        return self.valid_images[:idx], self.valid_image_classes[:idx]

    def test(self, batch_size=None):
        idx = len(self.test_images) if batch_size is None else batch_size
        return self.test_images[:idx], self.test_image_classes[:idx]

    def reset(self):
        if type(self.train_images_iteration_technique) is SequentialIterationTechnique:
            self.train_images_iteration_technique.reset(self.train_images)

    def data_shape(self):
        return (28, 28)