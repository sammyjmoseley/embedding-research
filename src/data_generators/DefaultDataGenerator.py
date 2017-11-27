class AbstractTripletTechnique(object):
    def next_triplet(self):
        raise BaseException("not implemented")


'''
reference_img != positive_img
reference_aug == positive_aug
'''


class AugmentationTripletTechnique(AbstractTripletTechnique):
    def next_triplet(self):
        return [None, None, None], [None, None, None], None


class AbstractIterationTechnique(object):
    def next(self):
        raise BaseException("not implemented")

    def reset(self):
        raise BaseException("not implemented")


class RandomIterationTechnique(AbstractIterationTechnique):
    def next(self):
        return None

    def reset(self):
        pass


class TripletDataset(object):
    def __init__(self, r, p, n, r_class, p_class, n_class, weights):
        self.r = r
        self.p = p
        self.n = n
        self.r_class = r_class
        self.p_class = p_class
        self.n_class = n_class
        self.weights = weights

    def get_reference(self):
        return None

    def get_positive(self):
        return None

    def get_negative(self):
        return None

    def get_reference_class(self):
        return None

    def get_positive_class(self):
        return None

    def get_negative_class(self):
        return None

    def get_weights(self):
        return None


class RotatedMNISTDataGenerator(object):
    def __init__(self,
                 train_ratio=0.85,
                 valid_ratio=0.05,
                 test_ratio=0.1,
                 triplet_technique=AugmentationTripletTechnique(),
                 train_iteration_technique=RandomIterationTechnique()):
        pass


    def train(self, batch_size):
        return None, None

    def triplet_train(self, batch_size):
        return None

    def test(self, batch_size=None):
        return None, None

    def validation(self, batch_size=None):
        return None

    def reset(self):
        pass

    def data_shape(self):
        return (28, 28)