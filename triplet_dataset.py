import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import random
import math
from scipy.ndimage import interpolation

class MnistDataset(object):
    train_split = 0.8

    def __init__(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.train_images = []
        self.test_images = []

        images = []
        for image, label in zip(mnist.test.images, mnist.test.labels):
            images.append([label, np.reshape(image, (28,28,1))])
        images = np.array(images)
        idx = int(len(images) * self.train_split)
        self.train_images = images[:idx]
        self.test_images = images[idx:]

    def __generate_data(self, arr, batch_size):
        x = []
        xp = []
        xn = []
        for i in range(batch_size):
            # Draw a random class
            a = arr[[random.randint(0, len(arr) - 1) for _ in range(0, 3)]]
            c = np.random.choice([True, False], 1, replace=False)

            if c[0]:
                def trans_p(m):
                    return np.flip(m, 1)

                def trans_n(m):
                    return m
            else:
                def trans_p(m):
                    return m

                def trans_n(m):
                    return np.flip(m, 1)
            x.append(trans_p(a[0][1]))
            xp.append(trans_p(a[1][1]))
            xn.append(trans_n(a[2][1]))

        return np.array(x), np.array(xp), np.array(xn), np.ones((batch_size,))

    def generate_train_data(self, batch_size):
        return self.__generate_data(self.train_images, batch_size)

    def generate_test_data(self, batch_size):
        return self.__generate_data(self.test_images, batch_size)[0]

class MnistDatasetSmallRotations(object):
    train_split = 0.8

    def __init__(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.train_images = []
        self.test_images = []

        images = []
        for image, label in zip(mnist.test.images, mnist.test.labels):
            images.append([np.array([np.argmax(label)]), np.reshape(image, (28,28,1))])
        images = np.array(images)
        idx = int(len(images) * self.train_split)
        self.train_images = images[:idx]
        self.test_images = images[idx:]

    def __generate_data(self, arr, batch_size):
        x = []
        xp = []
        xn = []
        weights = []
        for i in range(batch_size):
            # Draw a random class
            a = arr[[random.randint(0, len(arr) - 1) for _ in range(0, 2)]]
            if a[0][0] == a[1][0]:
                i = i-1
                continue
            #theta = random.randint(-30, 30)
            theta = 0

            def augment(img):
                return interpolation.rotate(img, theta, reshape=False)

            #weights.append(math.pow((theta)/60, 2.0))
            x.append(a[0][1])
            label = a[0][0]
            indices = np.where(arr[:,0] == label)[0]
            xp.append(arr[np.random.choice(indices)][1])
            #xp.append(augment(a[0][1]))
            xn.append(a[1][1])

        return np.array(x), np.array(xp), np.array(xn), np.array(weights)

    def generate_train_data(self, batch_size):
        return self.__generate_data(self.train_images, batch_size)

    def generate_test_data(self, batch_size):
        return self.__generate_data(self.test_images, batch_size)[0]


if __name__ == '__main__':
    mnist_dataset = MnistDataset()
    mnist_dataset.generate_mnist_data(1)
