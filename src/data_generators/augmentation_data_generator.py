import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from functools import reduce
import random
from PIL import Image
import gzip
import _pickle as pickle
import os

from data_generators.abstract_data_generator import AbstractGenerator
from data_generators.triplet_dataset import TripletDataset

image_shape = (28, 28, 1)


def __rotator(old_img, rot_ang):
    img = np.reshape(np.uint8(old_img * 255), (28, 28))
    src_im = Image.fromarray(img)
    im = src_im.convert('RGBA')
    dst_im = Image.new("RGBA", img.shape, "black")
    rot = im.rotate(rot_ang, expand=1).resize((28, 28))
    dst_im.paste(rot, (0, 0), rot)
    data = np.matmul(np.asarray(dst_im), [1.0 / 3.0, 1.0 / 3.0, 1 / 3.0, 0]).reshape(image_shape)
    data = np.float64(data) / 255.0
    return data, np.array([rot_ang])

default_augmentations = [(lambda ang: lambda x: __rotator(x, ang))(i) for i in range(-30, 31, 15)]
def random_augmentation(img):
    angle = np.random.uniform(-30, 31)
    return __rotator(img, angle)



class AugmentationDataGenerator(AbstractGenerator):
    """
    augmentations (img) -> (augmented image, numpy vec repr)
    """
    def __init__(self,
                 augmentations=default_augmentations,
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

        self.full_images = []
        self.full_augs = []
        self.full_labels = []
        self.full_img_ids = []

        self.is_epochal = is_epochal

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
        img_ids = np.array(list(range(0, len(images))))

        # make it deterministic
        np.random.seed(1)
        random.seed(1)

        for i in range(images.shape[0]):
            images[i] = random_augmentation(images[i])[0]

        image_augmentor = lambda x: map(lambda f: f(x), augmentations)
        label_augmentor = lambda x: map(lambda f: x, augmentations)

        def augmentor(l, f):
            ret = map(f, l)
            ret = reduce(lambda x,y: list(x)+list(y), ret)
            ret = map(lambda x: np.array([x]), ret)
            ret = list(ret)
            return np.concatenate(ret)

        def get_tuple(idx):
            return lambda x: x[idx]

        augs = augmentor(images, lambda x: map(get_tuple(1), image_augmentor(x)))
        images = augmentor(images, lambda x: map(get_tuple(0), image_augmentor(x)))
        labels = augmentor(labels, label_augmentor)
        img_ids = augmentor(img_ids, label_augmentor)

        valid_ratio += train_ratio
        test_ratio += valid_ratio

        def random_idxs(size):
            l = list(range(0, size))
            np.random.shuffle(l)
            return l
        # shuffle the images, we make sure augs, labels, and ids are preserved in the same order
        rand_indxs = random_idxs(len(images))
        images = images[rand_indxs]
        augs = augs[rand_indxs]
        labels = labels[rand_indxs]
        img_ids = img_ids[rand_indxs]

        train_idx = int(len(images) * train_ratio)
        valid_idx = int(len(images) * valid_ratio)
        test_idx = int(len(images) * test_ratio)

        self.train_images = images[:train_idx]
        self.train_image_augs = augs[:train_idx]
        self.train_image_classes = labels[:train_idx]
        self.train_image_ids = img_ids[:train_idx]
        augs = np.sum(np.square(self.train_image_augs), axis=1)
        self.train_image_orig = [augs==0]

        self.valid_images = images[train_idx:valid_idx]
        self.valid_image_augs = augs[train_idx:valid_idx]
        self.valid_image_classes = labels[train_idx:valid_idx]

        self.test_images = images[valid_idx:test_idx]
        self.test_image_augs = augs[valid_idx:test_idx]
        self.test_image_classes = labels[valid_idx:test_idx]

        self.full_images = images
        self.full_augs = augs
        self.full_labels = labels
        self.full_img_ids = img_ids

        self.train_epochal = 0

    def __next_image(self):
        raise BaseException("not implemented")

    def train(self, batch_size, only_originals=False):
        actual_train_images = self.train_images
        actual_train_image_classes = self.train_image_classes
        if only_originals:
            actual_train_images = self.train_images[self.train_image_orig]
            actual_train_image_classes = self.train_image_classes[self.train_image_orig]

        if self.is_epochal:
            if self.train_epochal >= len(actual_train_images):
                self.reset()
            batch_size = min(batch_size, len(actual_train_images)-self.train_epochal)
            idxs = range(np.sum(self.train_epochal), len(actual_train_images))[:batch_size]
            self.train_epochal += batch_size
            return actual_train_images[idxs], actual_train_image_classes[idxs]
        else:
            idxs = random.sample(range(0, len(actual_train_images)), batch_size)
            return actual_train_images[idxs], actual_train_image_classes[idxs]

    def triplet_train(self, batch_size):
        if self.is_epochal:
            if self.train_epochal >= len(self.train_images):
                self.reset()
            batch_size = min(batch_size, len(self.train_images)-self.train_epochal)
            idxs = range(np.sum(self.train_epochal), len(self.train_images))[:batch_size]
            self.train_epochal += batch_size

        else:
            idxs = random.sample(range(0, len(self.train_images)), batch_size)

        ref_imgs = self.train_images[idxs]
        ref_augs = self.train_image_augs[idxs]
        ref_classes = self.train_image_classes[idxs]
        ref_img_ids = self.train_image_ids[idxs]

        func_clazzes = lambda eq: \
            lambda clazz: eq(np.argmax(self.train_image_classes, axis=1), np.argmax(clazz))

        func_ids = lambda eq: \
            lambda id: eq(self.train_image_ids, np.argmax(id))

        def tripleter(classes, func):

            choices = np.array(list(map(func, classes)))
            idxs_arr = np.array(list(range(0, len(self.train_images))))
            idxs_new = map(lambda x: np.random.choice(idxs_arr[x]), choices)
            idxs_new = np.array(list(idxs_new))
            pics = self.train_images[idxs_new]
            augs = self.train_image_augs[idxs_new]
            labels = self.train_image_classes[idxs_new]
            return pics, augs, labels

        eq = lambda x, y: x == y
        neq = lambda x, y: x != y
        pos_imgs, pos_augs, pos_classes = tripleter(ref_img_ids, func_ids(eq))
        neg_imgs, neg_augs, neg_classes = tripleter(ref_classes, func_clazzes(neq))

        weights = ref_augs - neg_augs
        weights = np.linalg.norm(weights, axis=1)

        return TripletDataset(r=ref_imgs,
                              p=pos_imgs,
                              n=neg_imgs,
                              r_class=ref_classes,
                              p_class=pos_classes,
                              n_class=neg_classes,
                              weights=weights)

    def validation(self, batch_size=None):
        batch_size = len(self.valid_images) if batch_size is None else min(len(self.valid_images), batch_size)
        if self.is_epochal:
            idxs = range(0, len(self.valid_images))[:batch_size]
            return self.valid_images[idxs], self.valid_image_classes[idxs]
        else:
            idxs = random.sample(range(0, len(self.valid_images)), batch_size)
            return self.valid_images[idxs], self.valid_image_classes[idxs]

    def test(self, batch_size=None, augment=True):
        batch_size = len(self.test_images) if batch_size is None else min(len(self.test_images), batch_size)
        if self.is_epochal:
            idxs = range(0, len(self.test_images))[:batch_size]
            return self.test_images[idxs], self.test_image_classes[idxs]
        else:
            idxs = random.sample(range(0, len(self.test_images)), batch_size)
            return self.test_images[idxs], self.test_image_classes[idxs]

    def reset(self):
        if self.is_epochal:
            shuffle_idx = np.random.permutation(len(self.train_images))
            self.train_images = self.train_images[shuffle_idx]
            self.train_image_augs = self.train_image_augs[shuffle_idx]
            self.train_image_classes = self.train_image_classes[shuffle_idx]
            self.train_image_ids = self.train_image_ids[shuffle_idx]
            augs = np.sum(np.square(self.train_image_augs), axis=1)
            self.train_image_orig = [augs==0]
            self.train_epochal = 0
        else:
            pass

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

    def data_shape(self):
        return (28, 28, 1)

file_location = "rotated_dataset.gz"
augmentation_clazz = AugmentationDataGenerator

"""
is_epochal None indicates used saved preference, boolean value overrides saved value
"""

def load_augmentation_data_generator(is_epochal=None):
    if os.path.exists(file_location):
        f = gzip.open(file_location, "rb")
        unpickler = pickle.Unpickler(f)
        unpickler.find_class("data_generators.augmentation_data_generator", "AugmentationDataGenerator")
        ret = unpickler.load()
        f.close()
        if is_epochal is not None:
            ret.is_epochal = is_epochal
        return ret
    else:
        if is_epochal is not None:
            rotation_augmentation = AugmentationDataGenerator(is_epochal=is_epochal)
        else:
            rotation_augmentation = AugmentationDataGenerator()
        f = gzip.open(file_location, "wb")
        pickle.dump(rotation_augmentation, f)
        f.close()
        return rotation_augmentation

if __name__ == "__main__":
    generator = load_augmentation_data_generator()
    v_embed = generator.get_embedding_visualization_data()
    print(v_embed[0].shape, v_embed[1].shape)