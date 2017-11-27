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
        return lambda x: x, lambda x: x, 0.0