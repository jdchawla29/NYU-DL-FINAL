import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import datetime
from PIL import Image

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import v2, InterpolationMode
from torchvision import utils as vutils
from torch import nn
from torch.nn import functional as F
from torchmetrics import JaccardIndex

# local imports
from utils.params import Params

class TrainSegmentationTransforms:
    """
    Class for applying transforms to the training data.
    """

    def __call__(self, image, mask, params = Params()):
        """
        Apply transforms to the image and mask.

        Args:
            image (Tensor): Image tensor.
            mask (Tensor): Mask tensor.
            params (Params): Parameters for the transforms.
        """

        image_transforms = v2.Compose([
            v2.ToDtype(params.frame_dtype, scale=True), # Convert the image to a PyTorch Tensor and scale pixel values to [0, 1]
            # v2.Lambda(lambda X: 2 * X - 1.0),  # rescale to [-1, 1]
            v2.Resize(params.resolution, interpolation=InterpolationMode.BICUBIC), # Resize the image to the specified resolution
            # v2.Normalize(mean=[0.50613105, 0.50451115, 0.50091129], std=[0.05694459, 0.05666152, 0.06111675]) # <----------------- should be based on train set stats
        ])

        mask_transforms = v2.Compose([
            v2.ToDtype(params.mask_dtype), # Convert the mask to a PyTorch Tensor
            v2.Resize(params.resolution, interpolation=InterpolationMode.NEAREST_EXACT), # Resize the mask to the specified resolution
        ])

        image = image_transforms(image)
        mask = mask_transforms(mask)

        return image, mask

class ValSegmentationTransforms:
    """
    Class for applying transforms to the validation data.
    """

    def __call__(self, image, mask, params = Params()):
        """
        Apply transforms to the image and mask.

        Args:
            image (Tensor): Image tensor.
            mask (Tensor): Mask tensor.
            params (Params): Parameters for the transforms.
        """

        image_transforms = v2.Compose([
            v2.ToDtype(params.frame_dtype, scale=True), # Convert the image to a PyTorch Tensor and scale pixel values to [0, 1]
            # v2.Lambda(lambda X: 2 * X - 1.0),  # rescale to [-1, 1]
            v2.Resize(params.resolution, interpolation=InterpolationMode.BICUBIC), # Resize the image to the specified resolution
            # v2.Normalize(mean=[0.50613105, 0.50451115, 0.50091129], std=[0.05694459, 0.05666152, 0.06111675]) # <----------------- should be based on val set stats
        ])

        mask_transforms = v2.Compose([
            v2.ToDtype(params.mask_dtype), # Convert the mask to a PyTorch Tensor
            v2.Resize(params.resolution, interpolation=InterpolationMode.NEAREST_EXACT), # Resize the mask to the specified resolution
        ])

        image = image_transforms(image)
        mask = mask_transforms(mask)

        return image, mask
    
class HiddenSetTransforms:
    """
    Class for applying transforms to the hidden set data.
    """

    def __call__(self, images, params = Params()):
        """
        Apply transforms to the image and mask.

        Args:
            images (list(Tensor)): List of the 11 image tensors.
            params (Params): Parameters for the transforms.
        """

        image_transforms = v2.Compose([
            v2.ToDtype(params.frame_dtype, scale=True), # Convert the image to a PyTorch Tensor and scale pixel values to [0, 1]
            # v2.Lambda(lambda X: 2 * X - 1.0),  # rescale to [-1, 1]
            # v2.Resize(params.resolution, interpolation=InterpolationMode.BICUBIC), # Resize the image to the specified resolution
        ])


        images = [image_transforms(image) for image in images]

        return images