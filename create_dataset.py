from PIL import Image
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# amount training images are rotated in file (wrp to x axis)
prerotation = {1: 90, 2: 0, 3: 45}
img_size = (28, 28)
img_shape = img_size + (1,)

src_im = Image.open('data/arrow2.png')
im = src_im.convert('RGBA')

def generate_data(size):
    x = []
    y = []
    angles = []
    for i in range(0, size):
        rot_ang = random.randrange(0, 360)
        dst_im = Image.new("RGBA", img_size, "white")
        rot = im.rotate(rot_ang, expand=1).resize((28, 28))
        dst_im.paste(rot, (0, 0), rot)
        data = np.matmul(np.asarray(dst_im), [1.0 / 3.0, 1.0 / 3.0, 1 / 3.0, 0]).reshape(img_shape)
        label = (rot_ang + prerotation[2]) % 360
        x.append(data)
        y.append([math.sin(math.radians(label)), math.cos(math.radians(label))])
        angles.append([label])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (x, y, angles)

src_im_circle = Image.open("data/circle.png")
im_circle = src_im_circle.convert('RGBA')

def generate_data_circle(size):
    x = []
    y = []
    angles = []
    r = 10
    for i in range(0, size):
        rot_ang = math.radians(random.randrange(0, 4)*90)
        dst_im = Image.new("RGBA", img_size, "white")
        rot = im_circle.resize((1, 1))
        dst_im.paste(rot, (r + int(round(math.cos(rot_ang)*r)), r +int(round(math.sin(rot_ang)*r))), rot)
        data = np.matmul(np.asarray(dst_im), [1.0 / 3.0, 1.0 / 3.0, 1 / 3.0, 0]).reshape(img_shape)
        x.append(data)
        y.append([math.sin(rot_ang), math.cos(rot_ang)])
        angles.append([rot_ang])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (x, y, angles)

def generate_data_angles(size, window_width=11):
    x = []
    y = []
    i = 0
    j = 0
    for _ in range(0, size):
        i += 1
        if i >= window_width:
            j += 1
            i = 0
        if j >= window_width:
            j = 0
        if i == window_width/2 and j== window_width/2:
            continue
        square = [0 for _ in range(0, window_width * window_width)]
        square[i + window_width*j] = 1
        x.append(square)
        o = float(j-window_width/2)
        a = float(i - window_width / 2)
        h = math.sqrt(math.pow(o, 2.0) + math.pow(a, 2.0))
        cos =  a/h
        sin = o/h
        y.append([cos, sin])
    return x, y



if (__name__ == '__main__'):

    img = Image.fromarray(generate_data_circle(1)[0][0].reshape(28, 28))
    img.show()
    # plt.gray()
    # print Image.fromarray(generate_data_circle(1)[0][0]).show()
