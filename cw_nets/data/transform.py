"""
Image transformation, augmentation, etc. for use in models.
-----------------------------------------------------------

Where possible, the codebase uses albumentations implementations for transforms
because they checked various different implementations and use the fastest one.
However, in some cases the albumentations implementation uses a cv2 backend,
which is incompatible with unusual channel counts in imagery, and therefore
other implementations are used for those functions here.

"""

import numpy as np
import albumentations.functional as F
from albumentations.core.transforms_interface import DualTransform, to_tuple
from PIL.Image import BICUBIC, BILINEAR, HAMMING, NEAREST, LANCZOS
from Pillow import Image
from scipy import ndimage as ndi


class Rotate(DualTransform):
    """Array rotation using scipy.ndimage's implementation.

    Arguments
    ---------
    limit : ``[int, int]`` or ``int``
        Range from which a random angle is picked. If only a single `int` is
        provided, an angle is picked from range(-angle, angle)
    border_mode : str, optional
        One of ``['reflect', 'nearest', 'constant', 'wrap']``. Defaults to
        ``'reflect'``. See :func:`scipy.ndimage.interpolation.rotate`
        ``mode`` argument.
    cval : int or float, optional
        constant value to fill borders with if ``border_mode=='constant'``.
        Defaults to 0.
    always_apply : bool, optional
        Apply this transformation to every image? Defaults to no (``False``).
    p : float [0, 1], optional
        Probability that the augmentation is performed to each image. Defaults
        to ``0.5``.

    """
    def __init__(self, limit=90, border_mode='reflect', cval=0.0,
                 always_apply=False, p=0.5):
        super(Rotate, self).__init__(always_apply, p)

        self.limit = to_tuple(limit)
        self.border_mode = border_mode
        self.cval = cval

    def apply(self, im_arr, angle=0, border_mode='reflect', cval=0, **params):
        return ndi.interpolation.rotate(im_arr, angle, self.border_mode,
                                        self.cval)

    def get_params(self):
        return {'angle': np.random.randint(self.limit[0], self.limit[1])}

    def apply_to_bbox(self, bbox, angle=0, **params):
        return F.bbox_rotate(bbox, angle, **params)

    def apply_to_keypoint(self):
        raise NotImplementedError


class RandomScale(DualTransform):
    """Randomly resize the input array in X and Y.

    Arguments
    ---------
    scale_limit : ``(float, float)`` tuple or float
        Limit to the amount of scaling to perform on the image. If provided
        as a tuple, the limits are
        ``[shape*scale_limit[0], shape*scale_limit[1]]``. If only a single
        vaue is passed, this is converted to a tuple by converting to
        ``(1-scale_limit, 1+scale_limit)``, i.e. ``scale_limit=0.2`` is
        equivalent to ``scale_limit=(0.8, 1.2)``.
    axis : str, optional
        Which axis should be rescaled? Options are
        ``['width', 'height', 'both'].``
    interpolation : str, optional
        Interpolation method to use for resizing. One of
        ``['bilinear', 'bicubic', 'lanczos', 'nearest', or 'hamming']``.
        Defaults to ``'bicubic'``. See the Pillow_ documentation for more
        information.
    always_apply : bool, optional
        Apply this transformation to every image? Defaults to no (``False``).
    p : float [0, 1], optional
        Probability that the augmentation is performed to each image. Defaults
        to ``0.5``.

    .. _: https://pillow.readthedocs.io/en/4.1.x/handbook/concepts.html#filters-comparison-table

    """
    def __init__(self, scale_limit, axis='both', interpolation='bicubic',
                 always_apply=False, p=0.5):
        super(RandomScale, self).__init__(always_apply, p)

        self.scale_limit = to_tuple(scale_limit)
        # post-processing to fix values if only a single number was passed
        self.axis = axis
        if self.scale_limit[0] == -self.scale_limit[1]:
            self.scale_limit = [self.scale_limit[0]+1, self.scale_limit[1]+1]
        if interpolation == 'bicubic':
            self.interpolation = BICUBIC
        elif interpolation == 'bilinear':
            self.interpolation = BILINEAR
        elif interpolation == 'lanczos':
            self.interpolation = LANCZOS
        elif interpolation == 'nearest':
            self.interpolation = NEAREST
        elif interpolation == 'hamming':
            self.interpolation = HAMMING
        else:
            raise ValueError(
                'The interpolation argument is not one of: ' +
                '["bicubic", "bilinear", "hamming", "lanczos", "nearest"]')

    def get_params(self):
        if self.axis == 'height':
            x = 1
            y = np.random.uniform(self.scale_limit[0], self.scale_limit[1])
        elif self.axis == 'width':
            x = np.random.uniform(self.scale_limit[0], self.scale_limit[1])
            y = 1
        elif self.axis == 'both':
            x = np.random.uniform(self.scale_limit[0], self.scale_limit[1])
            y = np.random.uniform(self.scale_limit[0], self.scale_limit[1])
        return {'scale_x': x, 'scale_y': y}

    def apply(self, img, scale_x=1, scale_y=1, **params):
        return scale(img, scale_x, scale_y, self.interpolation)

    def apply_to_bbox(self, bbox, **params):
        # per Albumentations, bbox coords are scale-invariant
        return bbox

    def apply_to_keypoint(self, keypoint):
        raise NotImplementedError


def scale(im, scale_x, scale_y, interpolation):
    """Scale an image using Pillow."""
    im_shape = im.shape
    y_size = int(scale_y*im_shape[0])
    x_size = int(scale_x*im_shape[1])
    return np.array(Image.fromarray(im).resize((x_size, y_size),
                                               interpolation))
