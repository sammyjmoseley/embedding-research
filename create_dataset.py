from PIL import Image
import numpy as np
import random
import tensorflow as tf
import math

# amount training images are rotated in file (wrp to x axis)
prerotation = {1: 90, 2: 0, 3: 45}
img_size = (28, 28)
img_shape = img_size + (1,)

src_im = Image.open('data/arrow2.png')
im = src_im.convert('RGBA')

def generate_data(size):
    x = []
    y = []
    for i in range(0, size):
        rot_ang = random.randrange(0, 360)
        dst_im = Image.new("RGBA", img_size, "white")
        rot = im.rotate(rot_ang, expand=1).resize((28, 28))
        dst_im.paste(rot, (0, 0), rot)
        data = np.matmul(np.asarray(dst_im), [1.0 / 3.0, 1.0 / 3.0, 1 / 3.0, 0]).reshape(img_shape)
        label = (rot_ang + prerotation[2]) % 360
        x.append(data)
        y.append([math.sin(math.radians(label)), math.cos(math.radians(label))])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = tf.contrib.data.Dataset.from_tensors(tf.constant(x))
    y = tf.contrib.data.Dataset.from_tensors(tf.constant(y))
    return tf.contrib.data.Dataset.zip((x, y))

print generate_data(10)
