# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import random
import numpy as np
try:
    import scipy.ndimage
except Exception:
    print("Scipy not supported!")


class DataAugmentation(object):
    """ Data Augmentation.

    Base class for applying common real-time data augmentation.

    This class is meant to be used as an argument of `input_data`. When training
    a model, the defined augmentation methods will be applied at training
    time only. Note that DataPreprocessing is similar to DataAugmentation,
    but applies at both training time and testing time.

    Arguments:
        None

    Parameters:
        methods: `list of function`. The augmentation methods to apply.
        args: A `list` of arguments list to use for these methods.

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
    """ Image Augmentation.

    Base class for applying real-time augmentation related to images.

    This class is meant to be used as an argument of `input_data`. When training
    a model, the defined augmentation methods will be applied at training
    time only. Note that ImagePreprocessing is similar to ImageAugmentation,
    but applies at both training time and testing time.

    Arguments:
        None.

    Parameters:
        methods: `list of function`. The augmentation methods to apply.
        args: A `list` of arguments list to use for these methods.

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

        Arguments:
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
        self.methods.append(self._random_flip_updown)
        self.args.append(None)

    def add_random_90degrees_rotation(self, rotations=[0, 1, 2, 3]):
        """ add_random_90degrees_rotation

        Randomly perform 90 degrees rotations.

        Arguments:
            rotations: `list`. Allowed 90 degrees rotations.

        Return:
             Nothing.

        """
        self.methods.append(self._random_90degrees_rotation)
        self.args.append([rotations])

    def add_random_rotation(self, max_angle=20.):
        """ add_random_rotation.

        Randomly rotate an image by a random angle (-max_angle, max_angle).

        Arguments:
            max_angle: `float`. The maximum rotation angle.

        Returns:
            Nothing.

        """
        self.methods.append(self._random_rotation)
        self.args.append([max_angle])

    def add_random_blur(self, sigma_max=5.):
        """ add_random_blur.

        Randomly blur an image by applying a gaussian filter with a random
        sigma (0., sigma_max).

        Arguments:
            sigma: `float` or list of `float`. Standard deviation for Gaussian
                kernel. The standard deviations of the Gaussian filter are
                given for each axis as a sequence, or as a single number,
                in which case it is equal for all axes.

        Returns:
            Nothing.

        """
        self.methods.append(self._random_blur)
        self.args.append([sigma_max])

    # --------------------------
    #  Augmentation Computation
    # --------------------------

    def _random_crop(self, batch, crop_shape, padding=None):
        oshape = np.shape(batch[0])
        if padding:
            oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
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

    def _random_90degrees_rotation(self, batch, rotations=[0, 1, 2, 3]):
        for i in range(len(batch)):
            num_rotations = random.choice(rotations)
            batch[i] = np.rot90(batch[i], num_rotations)

        return batch

    def _random_rotation(self, batch, max_angle):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                # Random angle
                angle = random.uniform(-max_angle, max_angle)
                batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle,
                                                              reshape=False)
        return batch

    def _random_blur(self, batch, sigma_max):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                # Random sigma
                sigma = random.uniform(0., sigma_max)
                batch[i] = \
                    scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
        return batch


class SequenceAugmentation(DataAugmentation):

    def __init__(self):
        raise NotImplementedError

    def random_reverse(self):
        raise NotImplementedError
