import os
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
    def __init__(self, data_dir, split, params = Params(), transforms = None, subset = 1):
        assert split in ['train', 'val'], 'Invalid type. Must be either train or validation.'
        self.root_dir = os.path.join(data_dir, split)
        self.split = split
        self.transforms = transforms
        self.params = params

        assert subset > 0 and subset <= 1, 'Invalid subset ratio. Must be between 0 and 1.'
        self.subset = subset # what percentage of the data to return 

        self.random_state = params.random_state

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


def get_video_segmentation_loaders(params = Params()):
    """
    Returns the training and validation data loaders for the video segmentation task.
    
    Args:
        data_dir (str): The directory where the data is stored.
        batch_size (int): The batch size for the data loaders.
        num_workers (int): The number of workers to use for loading the data.
        training_subset (float): The ratio of the training_subset data to use. Must be between 0 and 1.
        val_subset (float): The ratio of the validation data to use. Must be between 0 and 1.
        random_state (int): The random seed to use for subsetting the train/val data.
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

class HiddenDataSet(Dataset):
    """
    Returns the 11 frames for each video in the hidden set.
    """
    def __init__(self, data_dir, params = Params(), transforms = None):
        self.root_dir = os.path.join(data_dir, 'hidden')
        self.transforms = transforms
        self.params = params

        self.random_state = params.random_state

        self._load_data()

    def _load_data(self):
        
        print(f"Loading hidden data from {self.root_dir}")

        self.data = []

        video_folders = sorted(os.listdir(self.root_dir))
        
        for video_folder in video_folders:
            
            video_folder_path = os.path.join(self.root_dir, video_folder)
            
            frames = []
            # Read video frames and masks
            for i in range(11):
                frame_path = os.path.join(video_folder_path, f'image_{i}.png')
                frames.append(frame_path)
            
            self.data.append(frames)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = self.data[idx]

        images = [read_image(image) for image in images] # read_image returns a tensor of shape [C, H, W]

        if self.transforms is not None:
            images = self.transforms(images, self.params)
        else:
            images = [v2.ToDtype(torch.float32, scale=True)(image) for image in images]

        # convert the list of tensors to a single tensor
        images = torch.stack(images) # [11, C, H, W]

        return images

    def visualize_video(self, idx):
        """
        Visualize a specific video (11 frames) from the hidden set.
        
        Args:
            idx (int): The index of the example to visualize.
        """

        images = self.__getitem__(idx)

        images = list(torch.unbind(images, dim=0))

        print('Image 0 shape:', images[0].shape)

        # Convert the torch.tensor image to PIL for easy visualization
        images = [to_pil_image(image) for image in images]

        # Plotting
        _, ax = plt.subplots(1, 11, figsize=(12, 6))
        
        for i, image in enumerate(images):
            ax[i].imshow(image)
            ax[i].set_title(f'Hidden Video {idx}: Frame {i}')
            ax[i].axis('off')
        
        plt.show()

def get_hidden_set_loader(params = Params()):
    """
    Returns the data loader for the hidden set.
    
    Args:
        hidden_set_path (str): The path to the hidden set.
        batch_size (int): The batch size for the data loader.
        num_workers (int): The number of workers to use for loading the data.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        
    Returns:
        DataLoader: The hidden set data loader.
    """

    hidden_set_transforms = HiddenSetTransforms()

    hidden_set_data = HiddenDataSet(params.data_dir, params, hidden_set_transforms)
    hidden_set_loader = torch.utils.data.DataLoader(hidden_set_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    return hidden_set_loader