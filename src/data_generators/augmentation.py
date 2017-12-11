from PIL import Image
import random
import numpy as np
import math

class AbstractAugmentation(object):
    def __init__(self):
        pass

    def random_augmentation(self):
        raise BaseException("unimplemented")

    def random_single_augmentation(self):
        raise BaseException("unimplemented")


class RotationAugmentation(AbstractAugmentation):
    def __init__(self, ang_range=(0, 360)):
        self.ang_range = ang_range
    @staticmethod
    def rotator(img, rot_ang):
        img = np.reshape(np.uint8(img * 255), (28, 28))
        src_im = Image.fromarray(img)
        im = src_im.convert('RGBA')
        dst_im = Image.new("RGBA", img.shape, "black")
        rot = im.rotate(rot_ang, expand=1).resize((28, 28))
        dst_im.paste(rot, (0, 0), rot)
        data = np.matmul(np.asarray(dst_im), [1.0 / 3.0, 1.0 / 3.0, 1 / 3.0, 0]).reshape(img.shape)
        data = np.float64(data)/255.0
        return data

    def random_single_augmentation(self):
        rotator = RotationAugmentation.rotator

        return lambda x: rotator(x, random.randrange(*self.ang_range))

    def random_augmentation(self):
        def augmentor(r, p, n):
            rotator = RotationAugmentation.rotator
            rot_ang_p = random.randrange(*self.ang_range)
            # rot_ang_n = random.randrange(0, 360)
            rot_ang_n = 0
            delta = math.pow(rot_ang_p-rot_ang_n, 2.0)/math.pow(2.0*math.pi, 2.0)
            return rotator(r, rot_ang_p), rotator(p, rot_ang_p), rotator(n, rot_ang_n), delta

        return lambda r, p, n: augmentor(r, p, n)

    def random_hard_augmentation(self, loss_pos, loss_neg):
        def augmentor(r, ps, ns):
            rotator = RotationAugmentation.rotator
            rot_ang_p = random.randrange(*self.ang_range)
            # rot_ang_n = random.randrange(0, 360)
            rot_ang_n = 0
            delta = math.pow(rot_ang_p-rot_ang_n, 2.0)/math.pow(2.0*math.pi, 2.0)
            ref = rotator(r, rot_ang_p)
            loss_ps = np.array(map(lambda x: loss_pos(x, ref), map(lambda p: rotator(p, rot_ang_p), ps)))
            loss_ns = np.array(map(lambda x: loss_neg(x, ref), map(lambda n: rotator(n, rot_ang_n), ns)))

            max_loss_ps_idx = np.argmax(loss_ps)
            max_loss_ns_idx = np.argmax(loss_ns)
            return ref, rotator(ps[max_loss_ps_idx], rot_ang_p), rotator(ns[max_loss_ns_idx], rot_ang_n), delta

        return lambda r, ps, ns: augmentor(r, ps, ns)