# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import random
import numpy as np


class DataAugmentation(object):
    """ Data Augmentation.

    Base class for managing data augmentation.

    Arguments:
        None.

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
    """ ImageAugmentation.

    Augmentation methods designed especially for images.

    Arguments:
        None.

    """

    def __init__(self):
        super(ImageAugmentation, self).__init__()

    # ----------------------------
    #  Image Augmentation Methods
    # ----------------------------

    def add_random_crop(self, crop_shape, padding=None):
        """ add_random_crop.

        Randomly crop a picture according to 'crop_shape'. An optional padding
        can be specified, for padding picture with 0s (To conserve original
        image shape).

        Examples:
            ```python
            # Example: pictures of 32x32
            imgaug = tflearn.ImageAugmentation()
            # Random crop of 24x24 into a 32x32 picture => output 24x24
            imgaug.add_random_crop((24, 24))
            # Random crop of 32x32 with image padding of 6 (to conserve original image shape) => output 32x32
            imgaug.add_random_crop((32, 32), 6)
            ```

        Args:
            crop_shape: `tuple` of `int`. The crop shape (height, width).
            padding: `int`. If not None, the image is padded with 'padding' 0s.

        Returns:
            Nothing.

        """
        self.methods.append(self._random_crop)
        self.args.append([crop_shape, padding])

    def add_random_flip_leftright(self):
        """ add_random_flip_leftright.

        Randomly flip an image (left to right).

        Returns:
            Nothing.

        """
        self.methods.append(self._random_flip_leftright)
        self.args.append(None)

    def add_random_flip_updown(self):
        """ add_random_flip_leftright.

        Randomly flip an image (upside down).

        Returns:
            Nothing.

        """
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

    def _random_crop(self, batch, crop_shape, padding=None):
        oshape = np.shape(batch[0])
        new_batch = []
        npad = ((padding, padding), (padding, padding), (0, 0))
        for i in range(len(batch)):
            new_batch.append(batch[i])
            if padding:
                new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
            nh = random.randint(0, oshape[0] - crop_shape[0])
            nw = random.randint(0, oshape[1] - crop_shape[1])
            new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                        nw:nw + crop_shape[1]]
        return new_batch

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
