import numpy as np

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

if (__name__ == '__main__'):
    generate_data(1)
