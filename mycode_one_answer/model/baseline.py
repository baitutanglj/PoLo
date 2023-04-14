import torch


class baseline(object):

    def get_baseline_value(self):
        pass

    def update(self, target):
        pass


class ReactiveBaseline(baseline):
    def __init__(self, l):
        self.l = l
        self.b = torch.tensor(0.0)

    def get_baseline_value(self):
        return self.b

    def update(self, target):
        self.b = (1 - self.l) * self.b +self.l * target
