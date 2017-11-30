class AbstractGenerator(object):
    def __next_image(self):
        raise BaseException("not implemented")

    def train(self, batch_size):
        raise BaseException("not implemented")

    def triplet_train(self, batch_size):
        raise BaseException("not implemented")

    def test(self, batch_size=None, augment=True):
        raise BaseException("not implemented")

    def reset(self):
        raise BaseException("not implemented")

    def data_shape(self):
        raise BaseException("not implemented")