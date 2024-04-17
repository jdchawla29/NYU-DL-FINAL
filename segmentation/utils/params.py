import os
import torch
import pathlib

class Params:
    """
    Class that defines the default hyperparameters for the Segmentation model. This is modifiable by the user.
    """
    def __init__(self):
        self.lr = 0.0004
        self.batch_size = 128
        self.val_batch_size = 128
        self.resolution = (160, 240)
        self.data_dir = os.path.join(str(pathlib.Path(__file__).parent.resolve()).rsplit('/', 2)[0], 'data')
        self.max_epochs = 100
        self.train_subset = 1
        self.val_subset = 1
        self.num_workers = 4
        self.random_state = 42
        self.pin_memory = True
        self.num_frame_channels = 3
        self.num_classes = 49 # 0-48 classes (0 is the background class) (1 - 48 are the classes)
        self.frame_dtype = torch.float32
        self.mask_dtype = torch.long
        self.checkpointing = True
        self.checkpoint_path = os.path.join(str(pathlib.Path(__file__).parent.resolve()).rsplit('/', 1)[0], 'model_checkpoints')
        self.dpi = 300


    def __setattr__(self, __name, __value) -> None:
        """
        Set the value of a parameter in the Params object. This is if you want to dynamically create a new parameter.
        """
        self.__dict__[__name] = __value

    def return_dict(self):
        """
        Return the dictionary of the parameters, but converting the objects to strings.
        """
        return {k: str(v) for k, v in self.__dict__.items()}
