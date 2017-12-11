import random
from enum import Enum

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from data_generators.abstract_data_generator import AbstractGenerator
from data_generators.augmentation import RotationAugmentation
from data_generators.triplet_dataset import TripletDataset


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


class RotatedMNISTDataGenerator(AbstractGenerator):
    def __init__(self,
                 train_ratio=0.85,
                 valid_ratio=0.05,
                 test_ratio=0.1,
                 triplet_technique=TripletTechnique.AUGMENTATION,
                 augment=True):
        self.train_images = []
        self.valid_images = []
        self.test_images = []
        self.train_image_classes = []
        self.valid_image_classes = []
        self.test_image_classes = []
        self.triplet_technique = triplet_technique
        self.augmentor = RotationAugmentation(ang_range=(-30, 30))
        self.augment = augment
        self.idx = 0

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
        self.train_augmentations = self.__single_augment(self.train_images)

        self.valid_images = images[train_idx:valid_idx]
        self.valid_image_classes = labels[train_idx:valid_idx]

        self.test_images = images[valid_idx:test_idx]
        self.test_image_classes = labels[valid_idx:test_idx]

    def train(self, batch_size):
        length = self.train_images.shape[0]
        indices = self.idx%length, min(self.idx%length + batch_size, length)
        shuffle = True if self.idx%length + batch_size > length else False
        self.idx += batch_size
        if self.augment:
            ret = (self.__reshape(self.train_augmentations[indices[0]:indices[1]]),
                   self.__reshape(self.train_images[indices[0]:indices[1]])), \
                   self.train_image_classes[indices[0]:indices[1]]
        else:
            ret = self.__reshape(self.train_images[indices[0]:indices[1]]), self.train_image_classes[indices[0]:indices[1]]

        if shuffle:
            # idxs = np.array(list(range(0, length)))
            # np.random.shuffle(idxs)
            # self.train_images = self.train_images[idxs]
            # self.train_image_classes = self.train_image_classes[idxs]
            if self.augment:
                self.train_augmentations = self.__single_augment(self.train_images)

        return ret

    def validation(self, batch_size=None):
        random.seed(a=1)
        idx = len(self.valid_images) if batch_size is None else batch_size
        valid_images = self.valid_images[:idx]
        if self.augment:
            valid_images = self.__single_augment(valid_images), \
                           self.__reshape(valid_images)
        else:
            valid_images = self.__reshape(valid_images)
        random.seed(a=None)
        return valid_images, self.valid_image_classes[:idx]

    def test(self, batch_size=None, augment=True):
        random.seed(a=20)
        idx = len(self.test_images) if batch_size is None else batch_size
        test_images = self.test_images[:idx]
        if augment:
            test_images = self.__single_augment(test_images), \
                          test_images
        random.seed(a=None)
        return test_images, self.test_image_classes[:idx]

    def data_shape(self):
        return 28, 28, 1

    def __single_augment(self, images):
        new_shape = (1, 28, 28, 1)
        f = lambda x: np.reshape(self.augmentor.random_single_augmentation()(x), new_shape)
        images = np.concatenate(list(map(f, images)))
        return images

    def __reshape(self, images):
        new_shape = (1, 28, 28, 1)
        f = lambda x: np.reshape(x, new_shape)
        images = np.concatenate(list(map(f, images)))
        return images


if __name__ == "__main__":
    generator = RotatedMNISTDataGenerator()
    print(generator.triplet_train(16).get_reference().shape)