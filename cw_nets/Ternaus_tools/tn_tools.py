from TernausNetV2.models.ternausnet2 import TernausNetV2
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
import cv2
from skimage.morphology import watershed
import scipy.ndimage as ndimage
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

def get_model(model_path):
    model = TernausNetV2(num_classes=2)
    state = torch.load(model_path)
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}

    model.load_state_dict(state)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()
    return model

def pad(img, pad_size=32):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]

def minmax(img):
    out = np.zeros_like(img).astype(np.float32)
    if img.sum() == 0:
        return out

    for i in range(img.shape[2]):
        c = img[:, :, i].min()
        d = img[:, :, i].max()

        t = (img[:, :, i] - c) / (d - c)
        out[:, :, i] = t
    return out.astype(np.float32)

def reform_tile(tile, rollaxis=True):
    
    if rollaxis:
        tile = np.rollaxis(tile, 0,3)
    rgb = minmax(tile[:,:,(5,3,2)]) 
    
    tf = tile.astype(np.float32)/ (2**11 - 1)
    
    return np.concatenate([rgb, tf], axis=2) * (2**8 - 1)

def label_watershed(before, after, component_size=20):
    print("ndimage")
    markers = ndimage.label(after)[0]
    print('watershed started')
    labels = watershed(-before, markers, mask=before, connectivity=8)
    unique, counts = np.unique(labels, return_counts=True)

    for (k, v) in tqdm(dict(zip(unique, counts)).items()):
        if v < component_size:
            labels[labels == k] = 0
    return labels

def get_img_transform():
    img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406, 0, 0, 0, 0, 0, 0, 0, 0], 
                  std=[0.229, 0.224, 0.225, 1, 1, 1, 1, 1, 1, 1, 1])
    ])
    
    return img_transform

def predict(model, input_img, pads):
    
    prediction = F.sigmoid(model(input_img)).data[0].cpu().numpy()
    mask = (prediction[0] > 0.5).astype(np.uint8)
    contour = (prediction[1])
    seed = ((mask * (1 - contour)) > 0.5).astype(np.uint8)
    labels = label_watershed(mask, seed)
    labels = unpad(labels, pads).astype(np.uint8)
    
    predictDict = {'mask': mask,
                  'contour': contour,
                  'seed': seed,
                  'labels': labels}
    
    
    return predictDict
    