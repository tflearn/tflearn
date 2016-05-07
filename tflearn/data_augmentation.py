# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import random
import numpy as np


class DataAugmentation(object):
    """ Data Augmentation.

    A class that manage data augmentation.

    """

    def __init__(self):
        self.methods = []
        self.args = []

    def apply(self, batch):
        for i, m in enumerate(self.methods):
            if self.args[i]:
                batch = m(batch, *self.args[i])
            else:
                batch = m(batch)
        return batch


class ImageAugmentation(DataAugmentation):

    def __init__(self):
        super(ImageAugmentation, self).__init__()

    # ----------------------------
    #  Image Augmentation Methods
    # ----------------------------

    def add_random_crop(self):
        self.methods.append(self._random_crop)
        self.args.append(None)

    def add_random_flip_leftright(self):
        self.methods.append(self._random_flip_leftright)
        self.args.append(None)

    def add_random_flip_updown(self):
        self.methods.append(self._random_flip_leftright)
        self.args.append(None)

    def add_random_rotation(self):
        self.methods.append(self._random_rotation)
        self.args.append(None)

    def add_random_blur(self):
        self.methods.append(self._random_blur)
        self.args.append(None)

    # --------------------------
    #  Augmentation Computation
    # --------------------------

    def _random_crop(self, batch):
        raise NotImplementedError

    def _random_flip_leftright(self, batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

    def _random_flip_updown(self, batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.flipud(batch[i])
        return batch

    def _random_rotation(self, batch):
        raise NotImplementedError

    def _random_blur(self, batch):
        raise NotImplementedError


class SequenceAugmentation(DataAugmentation):

    def __init__(self):
        raise NotImplementedError

    def random_reverse(self):
        raise NotImplementedError
