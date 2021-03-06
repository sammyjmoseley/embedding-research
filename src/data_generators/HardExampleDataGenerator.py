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
                 train_iteration_technique=RandomIterationTechnique):
        self.train_images = []
        self.valid_images = []
        self.test_images = []
        self.train_image_classes = []
        self.valid_image_classes = []
        self.test_image_classes = []
        self.triplet_technique = triplet_technique
        self.train_images_iteration_technique = train_iteration_technique()
        self.augmentor = RotationAugmentation()

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
            return self.__single_augment(self.train_images[indices]), self.train_image_classes[indices]

        elif type(self.train_images_iteration_technique) is SequentialIterationTechnique:
            mask = self.train_images_iteration_technique.mask
            size = len(mask)
            idxs = np.array(range(0, size))[mask]
            if len(idxs) == 0:
                return None
            idxs = idxs[:max(len(idxs), batch_size)]
            mask[idxs] = False
            return self.__single_augment(self.train_images[idxs]), self.train_image_classes[idxs]

        else:
            raise BaseException('unknown iteration technique')

    def triplet_train(self, batch_size, fun_pos, fun_neg):
        if self.triplet_technique == TripletTechnique.AUGMENTATION:
            references = []
            positives = []
            negatives = []

            distribution = np.random.choice(10, batch_size)

            for i in range(0, 9):
                count = np.sum(distribution == i)

                pos_mask = (np.argmax(self.train_image_classes, axis=1) == i)
                neg_mask = (np.argmax(self.train_image_classes, axis=1) != i)

                pos_size = pos_mask.size
                neg_size = neg_mask.size

                pos_idxs = np.array(range(0, pos_size))[pos_mask]
                neg_idxs = np.array(range(0, neg_size))[neg_mask]

                pos_batch_size = int(min(pos_size, count))
                neg_batch_size = int(min(neg_size, count))

                batch_size = min(pos_batch_size, neg_batch_size)

                if batch_size == 0:
                    continue
                ref_random_idx = np.random.choice(pos_idxs, batch_size)

                pos_random_idxs = np.random.choice(pos_idxs, batch_size*2)
                neg_random_idxs = np.random.choice(neg_idxs, batch_size)

                references.append(pos_random_idxs[:int(batch_size)])
                positives.append(pos_random_idxs[int(batch_size):])
                negatives.append(neg_random_idxs)

            references = np.concatenate(references)
            positives = np.concatenate(positives)
            negatives = np.concatenate(negatives)
            reference_images = self.train_images[references]
            reference_classes = self.train_image_classes[references]
            positive_images = self.train_images[positives]
            positive_classes = self.train_image_classes[positives]
            negative_images = self.train_images[negatives]
            negative_classes = self.train_image_classes[negatives]

            images = np.array([reference_images, positive_images, negative_images])
            aug_images = map(lambda x: self.augmentor.random_augmentation()(*x), zip(reference_images, positive_images, negative_images))
            aug_images = list(aug_images)

            ns = (1, 28, 28, 1)
            reference_images = np.concatenate(list(map(lambda x: np.reshape(x[0], ns), aug_images)))
            positive_images = np.concatenate(list(map(lambda x: np.reshape(x[1], ns), aug_images)))
            negative_images = np.concatenate(list(map(lambda x: np.reshape(x[2], ns), aug_images)))
            weights = np.array(list(map(lambda x: x[3], aug_images)))

            return TripletDataset(r=reference_images,
                                  p=positive_images,
                                  n=negative_images,
                                  r_class=reference_classes,
                                  p_class=positive_classes,
                                  n_class=negative_classes,
                                  weights=weights)

        elif self.triplet_technique == TripletTechnique.IMAGE_AUGMENTATION:
            raise BaseException('not implemented')

    def validation(self, batch_size=None):
        random.seed(a=1)
        idx = len(self.valid_images) if batch_size is None else batch_size
        valid_images = self.valid_images[:idx]
        valid_images = self.__single_augment(valid_images)
        random.seed(a=None)
        return valid_images, self.valid_image_classes[:idx]

    def test(self, batch_size=None, augment=True):
        random.seed(a=20)
        idx = len(self.test_images) if batch_size is None else batch_size
        test_images = self.test_images[:idx]
        if augment:
            test_images = self.__single_augment(test_images)
        random.seed(a=None)
        return test_images, self.test_image_classes[:idx]

    def reset(self):
        if type(self.train_images_iteration_technique) is SequentialIterationTechnique:
            self.train_images_iteration_technique.reset(self.train_images)

    def data_shape(self):
        return 28, 28, 1

    def __single_augment(self, images):
        new_shape = (1, 28, 28, 1)
        f = lambda x: np.reshape(self.augmentor.random_single_augmentation()(x), new_shape)
        images = np.concatenate(list(map(f, images)))
        return images

    def __embedding_visualization_helper(self, clazz):
        viable_images = (np.argmax(self.full_labels, axis=1) == clazz)
        img_id = self.full_img_ids[viable_images][0]
        idxs = self.full_img_ids == img_id
        return self.full_images[idxs], self.full_labels[idxs]

    def get_embedding_visualization_data(self, classes=range(0, 10)):
        ret = [self.__embedding_visualization_helper(clazz) for clazz in classes]
        ret_img = [x for (x, _) in ret]
        ret_lbl = [y for (_, y) in ret]
        ret_img = np.concatenate(ret_img)
        ret_lbl = np.concatenate(ret_lbl)
        return ret_img, ret_lbl

if __name__ == "__main__":
    generator = RotatedMNISTDataGenerator()
    print(generator.triplet_train(16).get_reference().shape)