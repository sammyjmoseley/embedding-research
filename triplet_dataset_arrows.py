import numpy as np
from PIL import Image
from skimage import io
from skimage.transform import resize, rotate
import random

img_size = (28, 28)
img_shape = img_size + (1,)
classes = ["data/arrow2.png", "data/arrow3.jpg"]

def generate_data(batch_size):
    x = []
    xp = []
    xn = []
    for i in range(batch_size):
        # Draw a random class
        c = np.random.choice(classes, 2, replace=False)
        xi = io.imread(c[0], as_grey=True)
        xi = resize(xi, (28, 28))
        xni = io.imread(c[1], as_grey=True)
        xni = resize(xni, (28, 28))
        rot_ang = random.randrange(0, 360)
        xpi = rotate(xi, rot_ang)
        x.append(xi[..., np.newaxis])
        xn.append(xni[..., np.newaxis])
        xp.append(xpi[..., np.newaxis])
    return (x, xp, xn)

def generate_test(size):
    x = []
    for i in range(size):
        # Draw a random class
        c = np.random.choice(classes, 1)
        xi = io.imread(c[0], as_grey=True)
        xi = resize(xi, (28, 28))
        rot_ang = random.randrange(0, 360)
        xi = rotate(xi, rot_ang)
        x.append(xi[..., np.newaxis])
    return np.array(x)

if (__name__ == '__main__'):
    generate_data(1)
