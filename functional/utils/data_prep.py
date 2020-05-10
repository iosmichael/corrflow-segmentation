import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

M = 8

def image_loader(path):
    image = cv2.imread(path)
    image = np.float32(image) / 255.0
    image = cv2.resize(image, (256, 256))
    return image

def rgb_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(image)

def rgb_preprocess_jitter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image = transforms.ColorJitter(0.1,0.1,0.1,0.1)(image)
    image = transforms.ToTensor()(image)
    return image

def greyscale_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lightness = image[None,:,:,0]
    processed = lightness / 50 - 1  # 0 mean
    return torch.Tensor(processed)

def quantized_color_preprocess(image, centroids):
    h, w, c = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    ab = image[:,:,1:]

    a = np.argmin(np.linalg.norm(centroids[None, :, :] - ab.reshape([-1,2])[:, None, :], axis=2),axis=1)
    # 256 256  quantized color (4bit)

    quantized_ab = a.reshape([h, w, -1])
    preprocess = transforms.ToTensor()
    return preprocess(quantized_ab)

def squeeze_index(image, index_list):
    for i, index in enumerate(index_list):
        image[image == index] = i
    return image

def r_loader(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def a_loader(path):
    anno, _ = davis_io.imread_indexed(path)
    return anno

def r_prep(image):
    image = np.float32(image) / 255.0

    h,w = image.shape[0], image.shape[1]
    if w%M != 0: image = image[:,:-(w % M)]
    if h%M != 0: image = image[:-(h % M),]

    return transforms.ToTensor()(image)

def a_prep(image):
    h,w = image.shape[0], image.shape[1]
    if w % M != 0: image = image[:,:-(w%M)]
    if h % M != 0: image = image[:-(h%M),:]
    image = np.expand_dims(image, 0)
    return torch.Tensor(image).contiguous().long()