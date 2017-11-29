from PIL import Image
import random
import numpy as np
import math


class AbstractAugmentation(object):
    def __init__(self):
        pass

    '''
    returns (img augmentor a, img augmentor b, delta)
    '''
    def random_augmentation(self):
        raise BaseException("unimplemented")


class RotationAugmentation(AbstractAugmentation):
    def random_augmentation(self):
        def rotator(img, rot_ang):
            img = np.reshape(np.uint8(img*255), (28, 28))
            src_im = Image.fromarray(img)
            im = src_im.convert('RGBA')
            dst_im = Image.new("RGBA", img.shape, "white")
            rot = im.rotate(rot_ang, expand=1).resize((28, 28))
            dst_im.paste(rot, (0, 0), rot)
            data = np.matmul(np.asarray(dst_im), [1.0 / 3.0, 1.0 / 3.0, 1 / 3.0, 0]).reshape(img.shape)
            return data

        def augmentor(r, p, n):
            rot_ang_p = random.randrange(0, 360)
            rot_ang_n = random.randrange(0, 360)
            delta = math.pow(rot_ang_p-rot_ang_n, 2.0)/math.pow(2.0*math.pi, 2.0)
            return rotator(r, rot_ang_p), rotator(p, rot_ang_p), rotator(n, rot_ang_n), delta

        return lambda r, p, n: augmentor(r, p, n)