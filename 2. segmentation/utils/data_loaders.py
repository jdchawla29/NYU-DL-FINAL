import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# torch imports
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from torchvision.io import read_image
from torchvision.transforms import v2

# local imports
from utils.params import Params
from utils.data_transforms import TrainSegmentationTransforms, ValSegmentationTransforms, HiddenSetTransforms

class VideoSegmentationData(Dataset):
    """
    Returns one frame and its corresponding mask at a time for each video in the training or validation set.
    """
    def __init__(self, data_dir, split, params = Params(), transforms = None, subset = 1, get_last_frame_only = False):
        assert split in ['train', 'val'], 'Invalid type. Must be either train or validation.'
        self.root_dir = os.path.join(data_dir, split)
        self.split = split
        self.transforms = transforms
        self.params = params

        assert subset > 0 and subset <= 1, 'Invalid subset ratio. Must be between 0 and 1.'
        self.subset = subset # what percentage of the data to return 

        self.random_state = params.random_state
        self.get_last_frame_only = get_last_frame_only

        self._load_data()

    def _load_data(self):
        
        print(f"Loading {self.split} data from {self.root_dir}")

        self.data = []

        video_folders = sorted(os.listdir(self.root_dir))
        
        for video_folder in video_folders:
            
            video_folder_path = os.path.join(self.root_dir, video_folder)
            
            mask = np.load(os.path.join(video_folder_path, 'mask.npy'))

            # Read video frames and masks
            for i in range(22):
                frame_path = os.path.join(video_folder_path, f'image_{i}.png')

                if self.get_last_frame_only:
                    if i != 21:
                        continue

                try:
                    self.data.append((frame_path, mask[i]))
                except:
                    # If the mask is not available, create an empty mask and append it
                    self.data.append((frame_path, np.zeros(mask[0].shape))) 
                    print(frame_path + ' mask not available so using empty mask')

        np.random.seed(self.random_state)

        # Subset the data randomly if needed
        if self.subset < 1:
            indices = np.random.choice(len(self.data), int(self.subset * len(self.data)), replace=False)

            # Use these indices to select elements from the original array of tuples
            self.data = [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask = self.data[idx]

        image = read_image(image) # read_image returns a tensor of shape [C, H, W]
        mask = torch.tensor(mask[np.newaxis, :, :]) # [H, W] -> [1, H, W] with (C = 1)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask, self.params)
        else:
            image = v2.ToDtype(torch.float32, scale=True)(image)
            mask = v2.ToDtype(torch.float32)(mask)

        return image, mask

    def visualize_example(self, idx):
        """
        Visualize a specific training example and its corresponding label.
        
        Args:
            idx (int): The index of the example to visualize.
        """

        image, mask = self.__getitem__(idx)

        # print('Image shape:', image.shape)
        # print('Mask shape:', mask.shape)

        # Convert the torch.tensor image to PIL for easy visualization
        image = to_pil_image(image)

        # squeeze the mask to remove the channel dimension so that it can be visualized
        mask = mask.squeeze(0)

        # Plotting
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title('Input Image')
        ax[0].axis('off')
        
        ax[1].imshow(mask)
        ax[1].set_title('Mask')
        ax[1].axis('off')
        
        plt.show()

class HiddenDataSet(Dataset):
    """
    If params.reconstructed_img_dir is a directory, this class:
        A) Returns the predicted 22nd frame for each video. Assumes output from STDiff where there are folders named 'Pred_0', 'Pred_1', representing batches of predicted frames.
    Otherwise, we assume that we aren't using the reconstructed 22nd frames and this class:
        B) Returns the 11th frame for each video in the hidden set.
    """
    def __init__(self, data_dir, reconstructed_img_dir, params = Params(), transforms = None):
        self.root_dir = os.path.join(data_dir, 'hidden')
        self.transforms = transforms
        self.params = params
        self.reconstructed_img_dir = reconstructed_img_dir

        self._load_data()

    def _load_data(self):
        
        if self.reconstructed_img_dir is None:
            self.data = []

            video_folders = sorted(os.listdir(self.root_dir))

            for video_folder in video_folders:
                
                video_folder_path = os.path.join(self.root_dir, video_folder)
                frame_path = os.path.join(video_folder_path, f'image_10.png')

                self.data.append(frame_path)
        else:

            print(f"Loading hidden data from {self.reconstructed_img_dir}")

            self.data = []
            
            # Retrieve all folders starting with 'Pred_'
            folders = sorted(glob.glob(os.path.join(self.reconstructed_img_dir, 'Pred_*')), key=lambda x: int(x.split('_')[-1]))

            # List to hold file paths
            all_images = []

            # Loop through each folder
            for folder in folders:
                # Get all image files in the current folder, sorted numerically
                images = sorted(glob.glob(os.path.join(folder, 'img_*.png')), key=lambda x: int(x.split('_')[-1].split('.')[0]))
                all_images.extend(images)

            self.data = all_images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        image = read_image(image) # read_image returns a tensor of shape [C, H, W]

        if self.transforms is not None:
            image = self.transforms(image, self.params)
        else:
            image = v2.ToDtype(torch.float32, scale=True)(image)

        return image

    def visualize_example(self, idx):
        """
        Visualize a specific training example and its corresponding label.
        
        Args:
            idx (int): The index of the example to visualize.
        """

        image = self.__getitem__(idx) # [C, H, W]

        # Convert the torch.tensor image to PIL for easy visualization
        image = to_pil_image(image)

        # Plotting
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(image)
        ax.set_title('Hidden Set Image')
        ax.axis('off')
        
        plt.show()


def get_video_segmentation_loaders(params = Params()):
    """
    Returns the training and validation data loaders for the video segmentation task.
    
    Args (within the passed params object):
        data_dir (str): The directory where the data is stored.
        batch_size (int): The batch size for the data loaders.
        num_workers (int): The number of workers to use for loading the data.
        training_subset (float): The ratio of the training_subset data to use. Must be between 0 and 1.
        val_subset (float): The ratio of the validation data to use. Must be between 0 and 1.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        
    Returns:
        DataLoader: The training data loader.
        DataLoader: The validation data loader.
    """

    data_dir = params.data_dir
    batch_size = params.batch_size
    num_workers = params.num_workers
    training_subset = params.train_subset
    val_subset = params.val_subset
    pin_memory = params.pin_memory

    train_transforms = TrainSegmentationTransforms()
    val_transforms = ValSegmentationTransforms()

    train_data = VideoSegmentationData(data_dir, 'train', params, train_transforms, training_subset)
    val_data = VideoSegmentationData(data_dir, 'val', params, val_transforms, val_subset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def get_hidden_set_loader(params = Params()):
    """
    Returns the data loader for the hidden set.
    
    Args (within the passed params object):
        data_dir (str): The directory where the data is stored.
        num_workers (int): The number of workers to use for loading the data.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        
    Returns:
        DataLoader: The hidden set data loader.
    """

    data_dir = params.data_dir
    num_workers = params.num_workers
    pin_memory = params.pin_memory
    reconstructed_img_dir = params.reconstructed_img_dir

    hidden_set_transforms = HiddenSetTransforms()

    hidden_set_data = HiddenDataSet(data_dir, reconstructed_img_dir, params, hidden_set_transforms)
    hidden_set_loader = torch.utils.data.DataLoader(hidden_set_data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return hidden_set_loader