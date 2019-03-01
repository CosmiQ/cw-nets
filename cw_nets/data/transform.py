"""
Image transformation, augmentation, etc. for use in models.
-----------------------------------------------------------

Where possible, the codebase uses albumentations implementations for transforms
because they checked various different implementations and use the fastest one.
However, in some cases albumentations uses a cv2 backend,
which is incompatible with unusual channel counts in imagery, and therefore
other implementations are used for those functions here.

Note: Some augmentations are unavailable in this library. This is intentional.


Functionality used directly from albumentations:
- Crop
- VerticalFlip
- HorizontalFlip
- Flip
- Transpose
- Resize
- CenterCrop
- RandomCrop
- RandomSizedCrop
- OpticalDistortion
- GridDistortion
- ElasticTransform
- Normalize
- Cutout
- HueSaturationValue  # NOTE: CAN ONLY HANDLE RGB 3-CHANNEL!
- RGBShift  # NOTE: CAN ONLY HANDLE RGB 3-CHANNEL!
- RandomBrightnessContrast
- Blur
- MotionBlur
- MedianBlur
- GaussNoise
- CLAHE
- RandomGamma
- ToFloat

Implemented here:
- Rotate
- RandomScale
-

"""

import numpy as np
import albumentations.functional as F
from albumentations.functional import preserve_channel_dim
from albumentations.core.transforms_interface import DualTransform, to_tuple
from albumentations.core.transforms_interface import ImageOnlyTransform
from PIL.Image import BICUBIC, BILINEAR, HAMMING, NEAREST, LANCZOS, AFFINE
from Pillow import Image
from scipy import ndimage as ndi
import cv2


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


def Cutout(ImageOnlyTransform):
    """CoarseDropout of the square regions in the image.

    This is a slightly optimized version of the albumentations implementation.

    Arguments
    ---------
    num_holes : int
        number of regions to zero out
    h_size : int
        height of the hole
    w_size : int
        width of the hole

    Targets:
        image

    Image types:
        uint8, float32
    """
    def __init__(self, num_holes=8, max_h_size=8, max_w_size=8,
                 always_apply=False, p=0.5):
        super(Cutout, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def apply(self, image, **params):
        return F.cutout(image, self.num_holes,
                        self.max_h_size, self.max_w_size)


# NOTE ON THE ShiftScaleRotate CLASS BELOW:
# Aside from cv2, there is currently no good implementation that enables
# handling the border in any way other than filling. I'm not 100% sure this
# is going to work for >3-channel images but we're going to roll the dice for
# now.

# class ShiftScaleRotate(DualTransform):
#     """Shift/scale/rotate using Pillow.
#
#     Arguments
#     ---------
#     scale_limits : `int` or `tuple`, optional
#         Limits of re-scaling amount, in fraction of starting size. If an `int`
#         is passed, the image size will be rescaled randomly in the range
#         ``(1-scale_limits, 1+scale_limits)``. If a 2-tuple is provided, both
#         x and y scaling will have values drawn from
#         ``np.random.uniform(scale_limits[0], scale_limits[1])``. If a 4-tuple
#         is provided, x scaling will be selected from
#         ``np.random.uniform(scale_limits[0], scale_limits[1])``, and y scaling
#         will be selected from
#         ``np.random.uniform(scale_limits[2], scale_limits[3]).`` To maintain
#         aspect ratio, use `lock_aspect=True` (has no effect if a 4-tuple is
#         provided). If not passed, image is not scaled.
#     lock_aspect : bool, optional
#         Should aspect ratio be locked to the input ratio? Defaults to no
#         (``True``). If ``True`` and a 4-tuple is passed for `scale_limits`,
#         `lock_aspect` is ignored.
#     rotation_limits : `int` or 2-`tuple`, optional
#         Limit of degrees of rotation of the image. If an integer, then the
#         image will be rotated an angle randomly selected from
#         ``range(-rotation_limits, rotation_limits)`` degrees.
#         If a tuple of form ``(min_rotation, max_rotation)``, the image will be
#         rotated an angle randomly selected from
#         ``range(min_rotation, max_rotation)``. Defaults to 0 (no rotation).
#     translation_limits: int or 2-tuple, optional
#         Magnitude of x and y translations to perform in pixels. If a single int
#         is provided, both x and y translation will be selected from the random
#         uniform range of ``([-limit, +limit])``. If a 2-tuple
#         ``(x_limit, y_limit)`` is provided, the limits are set separately based
#         on the provided tuple.
#     interpolation : str, optional
#         Interpolation method to use for resizing. One of
#         ``['bilinear', 'bicubic', or 'nearest']``.
#         Defaults to ``'bicubic'``. See the Pillow_ documentation for more
#         information.
#     p : float [0, 1], optional
#         Probability that the augmentation is performed to each image. Defaults
#         to ``0.5``.
#
#     .. _: https://pillow.readthedocs.io/en/4.1.x/handbook/concepts.html#filters-comparison-table
#     """
#     def __init__(self, size=None, scale_limits=0, lock_aspect=True,
#                  rotation_limits=0, translation_limits=0,
#                  interpolation="bicubic", p=0.5, always_apply=False):
#         super(ShiftScaleRotate, self).__init__(always_apply, p)
#         self.lock_aspect = lock_aspect
#         if type(scale_limits) == float:
#             self.scale_limits = (1-scale_limits, 1+scale_limits,
#                                  1-scale_limits, 1+scale_limits)
#         elif type(scale_limits) == tuple:
#             if len(scale_limits) == 2:
#                 self.scale_limits = (scale_limits[0], scale_limits[1],
#                                      scale_limits[0], scale_limits[1])
#             elif len(scale_limits == 4):
#                 self.scale_limits = scale_limits
#                 self.lock_aspect = False  # force if a 4-tuple was provided
#             else:
#                 raise ValueError(
#                     "len(scale_limits) must be len 2 or 4 if it's a tuple.")
#         else:
#             raise TypeError('scale_limits must be a float or tuple.')
#         self.rotation_limits = to_tuple(rotation_limits)
#         self.translation_limits = to_tuple(translation_limits)
#         if interpolation == 'bicubic':
#             self.interpolation = BICUBIC
#         elif interpolation == 'bilinear':
#             self.interpolation = BILINEAR
#         elif interpolation == 'nearest':
#             self.interpolation = NEAREST
#         else:
#             raise ValueError(
#                 'The interpolation argument is not one of: ' +
#                 '["bicubic", "bilinear", "nearest"]')
#
#         def apply(self, img, angle=0, scale=0, dx=0, dy=0,
#                   interpolation=self.interpolation, **params):
#
#
#
#         def get_params(self):  # parameters to use in apply()
#             param_dict = {
#                 'angle': np.random.uniform(self.rotation_limits[0],
#                                            self.rotation_limits[1]),
#                 'xscale': np.random.uniform(1+self.scale_limits[0],
#                                             1+self.scale_limits[1]),
#                 'yscale': np.random.uniform(1+self.scale_limits[2],
#                                             1+self.scale_limits[3]),
#                 'dx': np.random.randint(self.translation_limits[0],
#                                         self.translation_limits[1]),
#                 'dy': np.random.randint(self.translation_limits[0],
#                                         self.translation_limits[1])}
#             if self.lock_aspect:  # reset param_dict['yscale'] to same as x
#                 param_dict['yscale'] = param_dict['xscale']
#             return param_dict


@preserve_channel_dim
def scale(im, scale_x, scale_y, interpolation):
    """Scale an image using Pillow."""
    im_shape = im.shape
    y_size = int(scale_y*im_shape[0])
    x_size = int(scale_x*im_shape[1])
    return np.array(Image.fromarray(im).resize((x_size, y_size),
                                               interpolation))


def cutout(img, num_holes, h_size, w_size):
    img = img.copy()
    height, width = img.shape[:2]
    ys = np.random.randint(0, height, num_holes)
    xs = np.random.randint(0, width, num_holes)
    y1s = np.clip((ys-h_size//2), 0, height)
    y2s = np.clip((ys+h_size//2), 0, height)
    x1s = np.clip((xs-h_size//2), 0, width)
    x2s = np.clip((xs+h_size//2), 0, width)
    for n in range(num_holes):
        img[y1s[n]:y2s[n], x1s[n]:x2s[n]] = 0
    return img

# see note above the ShiftScaleRotate class.
# @preserve_channel_dim
# def shift_scale_rotate(img, angle, xscale, yscale, dx, dy,
#                        interpolation=BICUBIC, border_mode='reflect'):
#         if type(img) != Image:
#             pillow_im = Image.fromarray(img)
#         else:
#             pillow_im = img
#         height, width = img.shape[:2]
#         center = (width/2, height/2)
#         #
#         matrix = cv2.getRotationMatrix2D(center, angle, scale)
#         matrix[0, 2] += dx
#         matrix[1, 2] += dy
#         pillow_im = pillow_im.transform(
#             size=(int(xscale*img.size[0]), int(yscale*img.size[1])), AFFINE,
#             data=matrix, resample=interpolation,)
