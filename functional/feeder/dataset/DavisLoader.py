import os
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random

import cv2
import numpy as np
import functional.utils.io as davis_io
from PIL import Image, ImageOps
from ...utils.data_prep import *

# for self-supervised learning
class DAVISColorizationDataset(data.Dataset):

    def __init__(self, annos, jpegs, training=True):
        self.jpegs = jpegs
        self.training = training
        self.p_2 = 0.1             # probability of random perturbation
        self.centroids = np.load('datas/centroids/centroids_16k_kinetics_10000samples.npy')

    def __getitem__(self, index):
        jpegs = self.jpegs[index]
        images = [image_loader(jpeg) for jpeg in jpegs]

        images_quantized = [quantized_color_preprocess(ref, self.centroids) for ref in images]

        r = np.random.random()
        if r < self.p_2:
            images_rgb = [rgb_preprocess_jitter(ref) for ref in images]
        else:
            images_rgb = [rgb_preprocess(ref) for ref in images]

        return images_rgb, images_quantized

    def __len__(self):
        return len(self.jpegs)

class DAVISSegmentationDataset(data.Dataset):

    def __init__(self, filepath, filenames, training):
        self.refs = filenames
        self.filepath = filepath
        self.training = training
        self.p_2 = 0.1             # probability of random perturbation
        self.centroids = np.load('datas/centroids/centroids_16k_kinetics_10000samples.npy', allow_pickle=True)

    def __getitem__(self, index):
        refs = self.refs[index]

        images = [image_loader(os.path.join(self.filepath, ref)) for ref in refs]

        images_quantized = [quantized_color_preprocess(ref, self.centroids) for ref in images]

        r = np.random.random()
        if r < self.p_2:
            images_rgb = [rgb_preprocess_jitter(ref) for ref in images]
        else:
            images_rgb = [rgb_preprocess(ref) for ref in images]

        return images_rgb, images_quantized

    def __len__(self):
        return len(self.refs)

