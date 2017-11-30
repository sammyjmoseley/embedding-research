import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from functools import reduce
import random

from data_generators.abstract_data_generator import AbstractGenerator


class AugmentationDataGenerator(AbstractGenerator):
    """
    augmentations (img) -> (augmented image, numpy vec repr)
    """
    def __init__(self,
                 augmentations,
                 train_ratio=0.85,
                 valid_ratio=0.05,
                 test_ratio=0.1,
                 is_epochal=False):
        self.train_images = []
        self.valid_images = []
        self.test_images = []

        self.train_image_augs = []
        self.valid_image_augs = []
        self.test_image_augs = []

        self.train_image_classes = []
        self.valid_image_classes = []
        self.test_image_classes = []
        self.is_epochal = False

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

        image_augmentor = lambda x: map(lambda f: f(x), augmentations)
        label_augmentor = lambda x: map(lambda f: x, augmentations)

        def augmentor(l, f):
            l = map(f, l)
            l = reduce(lambda x,y: x+y, l)
            l = list(l)
            return np.concatenate(l)

        def get_tuple(idx):
            return lambda x: x[idx]

        images = augmentor(images, lambda x: map(get_tuple(0), image_augmentor(x)))
        augs = augmentor(images, lambda x: map(get_tuple(1), image_augmentor(x)))
        labels = augmentor(labels, label_augmentor)

        valid_ratio += train_ratio
        test_ratio += valid_ratio

        train_idx = int(len(images) * train_ratio)
        valid_idx = int(len(images) * valid_ratio)
        test_idx = int(len(images) * test_ratio)

        self.train_images = images[:train_idx]
        self.train_image_augs = augs[:train_idx]
        self.train_image_classes = labels[:train_idx]

        self.valid_images = images[train_idx:valid_idx]
        self.valid_image_augs = augs[train_idx:valid_idx]
        self.valid_image_classes = labels[train_idx:valid_idx]

        self.test_images = images[valid_idx:test_idx]
        self.test_image_augs = augs[valid_idx:test_idx]
        self.test_image_classes = labels[valid_idx:test_idx]

        def random_idxs(size):
            l = list(range(0, size))
            np.random.shuffle(l)
            return l

        train_randomizer = random_idxs(len(self.train_images))
        self.train_images = self.train_images[train_randomizer]
        self.train_image_augs = self.train_image_augs[train_randomizer]
        self.train_image_classes = self.train_image_classes[train_randomizer]

        valid_randomizer = random_idxs(len(self.valid_images))
        self.valid_images = self.valid_images[valid_randomizer]
        self.valid_image_classes = self.valid_image_classes[valid_randomizer]
        self.valid_image_classes = self.valid_image_classes[valid_randomizer]

        test_randomizer = random_idxs(len(self.test_images))
        self.test_images = self.test_images[test_randomizer]
        self.test_image_classes = self.test_image_classes[test_randomizer]
        self.test_image_classes = self.test_image_classes[test_randomizer]

    def __next_image(self):
        raise BaseException("not implemented")

    def train(self, batch_size):
        if self.is_epochal:
            raise BaseException("not implemented")
        else:
            idxs = random.choice(range(0, len(self.train_images)), batch_size)
            return self.train_images[idxs], self.train_image_classes[idxs]

    def triplet_train(self, batch_size):
        if self.is_epochal:
            raise BaseException("not implemented")
        else:
            idxs = random.choice(range(0, len(self.train_images)), batch_size)
            ref_imgs = self.train_images[idxs]
            ref_augs = self.train_image_augs[idxs]
            ref_classes = self.train_image_classes[idxs]

            def tripleter(classes, eq):
                func = lambda clazz: eq(self.train_image_classes, clazz)
                choices = map(func, classes)
                idxs = np.array(list(range(0, len(self.train_images))))
                idxs = map(lambda x: np.random.choice(idxs[x]), choices)
                pics = self.train_images[idxs]
                augs = self.train_image_augs[idxs]
                labels = self.train_images[idxs]
                return pics, augs, labels

            pos_imgs, pos_augs, pos_classes = tripleter(ref_classes, lambda x, y: x == y)
            neg_imgs, neg_augs, neg_classes = tripleter(ref_classes, lambda x, y: x != y)


    def test(self, batch_size=None, augment=True):
        if self.is_epochal:
            raise BaseException("not implemented")
        else:
            pass

    def reset(self):
        if self.is_epochal:
            raise BaseException("not implemented")
        else:
            pass

    def data_shape(self):
        if self.is_epochal:
            raise BaseException("not implemented")
        else:
            pass