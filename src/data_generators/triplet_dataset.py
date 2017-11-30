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
        return self.r

    def get_positive(self):
        return self.p

    def get_negative(self):
        return self.n

    def get_reference_class(self):
        return self.r_class

    def get_positive_class(self):
        return self.p_class

    def get_negative_class(self):
        return self.n_class

    def get_weights(self):
        return self.weights

    def __iter__(self):
        return zip(zip(self.r, self.p, self.n), zip(self.r_class, self.p_class, self.n_class), self.weights)