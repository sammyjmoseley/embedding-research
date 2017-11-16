import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


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

    def __init__(self):
        def add_to_dataset(images, labels, image_map, labels_list):
            for img, lbl in zip(images, labels):
                if lbl not in self.train_images:
                    image_map[lbl] = []
                    labels_list.append(lbl)
                image_map[lbl].append(img)


        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.train_images = {}
        self.train_labels = []
        self.test_images = {}
        self.test_labels = []



        for image, label in zip(mnist.test.images, mnist.test.labels):
            if label not in self.test_images:



    def generate_mnist_data(batch_size):
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



if (__name__ == '__main__'):
    generate_data(1)
