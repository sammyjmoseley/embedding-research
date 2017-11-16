import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import random


img_size = (28, 28)
img_shape = img_size + (1,)
classes = [0, 0.5, 1]




def generate_data(batch_size):
    x = []
    xp = []
    xn = []
    for i in range(batch_size):
        # Draw a random class
        c = np.random.choice(classes, 2, replace=False)
        xi = np.zeros(img_shape)
        xi[:] = c[0]
        xni = np.zeros(img_shape)
        xni[:] = c[1]
        xpi = np.random.normal(c[0], scale=0.01, size=img_shape)
        xpi = np.minimum(1, np.maximum(0, xpi))
        x.append(xi)
        xn.append(xni)
        xp.append(xpi)
    return (x, xp, xn)

def generate_test(size):
    x = []
    for i in range(size):
        # Draw a random class
        c = np.random.choice(classes, 1)
        xi = np.random.normal(c[0], scale=0.01, size=img_shape)
        xi = np.minimum(1, np.maximum(0, xi))
        x.append(xi)
    return np.array(x)

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

        return np.array(x), np.array(xp), np.array(xn)

    def generate_train_data(self, batch_size):
        return self.__generate_data(self.train_images, batch_size)

    def generate_test_data(self, batch_size):
        return self.__generate_data(self.test_images, batch_size)




if (__name__ == '__main__'):
    mnist_dataset = MnistDataset()
    mnist_dataset.generate_mnist_data(1)
