# general imports
import os
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

# torch imports
import torch
from torchmetrics import JaccardIndex as IoU
from torch.optim import Adam
import torch.nn.functional as F

# local imports
from utils.params import Params
from utils.data_transforms import ValSegmentationTransforms, TrainSegmentationTransforms
from torchvision.transforms import v2, InterpolationMode
from utils.data_loaders import VideoSegmentationData
from utils.visualize import visualize_predicted_example
from models.unet import UNet

@torch.inference_mode()
def visualize_example(data_loader, model, split, device, epoch, params = Params()): 
    image_ex, true_mask_ex = next(iter(data_loader))
    
    model.eval()
    
    image_ex = image_ex.to(device)
    true_mask_ex = true_mask_ex.squeeze(1).to(device)[0]

    predicted_masks = model(image_ex)
    mask_pred_argmax = torch.argmax(F.softmax(predicted_masks, dim=1), dim=1).squeeze(0)[0]

    example_iou = calculate_iou(predicted_masks[0][np.newaxis, :, :], true_mask_ex[np.newaxis, :, :], params.num_classes, device)
    
    visualize_predicted_example(image_ex[0].detach().cpu(), true_mask_ex.detach().cpu(), mask_pred_argmax.detach().cpu(), f"True vs Predicted Mask ({split.capitalize()} Set) (IoU = {example_iou})", os.path.join(params.checkpoint_dir, f"{split}_{epoch}.png"), params)
    
    model.train()

    return



def calculate_iou(pred_masks, true_masks, num_classes, device):
    # Initialize the IoU metric
    IoU_func = IoU(task="multiclass", num_classes=num_classes).to(device)

    # Apply softmax to masks_pred to get probabilities
    masks_pred_softmax = F.softmax(pred_masks, dim=1)

    # Calculate Jaccard Index
    mask_pred_argmax = torch.argmax(masks_pred_softmax, dim=1)

    IoU_metric = IoU_func(mask_pred_argmax, true_masks).item()

    return IoU_metric


@torch.inference_mode()
def validate(model, val_loader, criterion, device, params = Params()):

    print("\nValidating...\n")

    model.eval()

    val_epoch_loss = 0
    val_epoch_iou = 0

    for i, (images, true_masks) in enumerate(tqdm(val_loader, desc=f"Model Validation")):

        images = images.to(device)
        true_masks = true_masks.squeeze(1).to(device)

        pred_masks = model(images)
        loss = criterion(pred_masks, true_masks)

        val_epoch_iou += calculate_iou(pred_masks, true_masks, params.num_classes, device)
        # loss += calculate_dice_loss(pred_masks, true_masks, params.num_classes)

        val_epoch_loss += loss.item()

    model.train()

    print()

    return val_epoch_iou / len(val_loader), val_epoch_loss / len(val_loader)



def train(model, train_loader, val_loader, criterion, optimizer, device, params = Params()):

    # create a custom checkpoint dir in the checkpoints directory if checkpointing is enabled for this run
    if params.checkpointing:
        
        # create the checkpoint directory for this run with the current timestamp
        checkpoint_dir = os.path.join(params.checkpoint_path, str(datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_"))
        os.makedirs(checkpoint_dir, exist_ok = True)

        # update the checkpoint path in the params object
        params.checkpoint_dir = checkpoint_dir

        # write the params to the checkpoint directory so that we can refer to them later
        with open(os.path.join(checkpoint_dir, 'params.json'), 'w') as f:
            json.dump(params.return_dict(), f, indent=2)

        print(f"Checkpointing enabled. Checkpoints will be saved to {checkpoint_dir}")

    # Set the model to training mode
    model.train()

    # create a file called log.txt in the checkpoint directory to log the training progress
    with open(os.path.join(params.checkpoint_dir, 'log.txt'), 'w') as f:

        best_val_iou = 0

        # Iterate over the epochs
        for epoch in range(params.max_epochs):

            train_epoch_loss = 0
            train_epoch_iou = 0

            # Iterate over the training data
            for i, (images, true_masks) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):

                # Move the images and masks to the device
                images = images.to(device)
                true_masks = true_masks.squeeze(1).to(device) # Remove the channel dimension - shape: (batch_size, 1, H, W) -> (batch_size, H, W)

                # Forward pass
                pred_masks = model(images)

                # Calculate the loss
                loss = criterion(pred_masks, true_masks)

                # Calculate the IoU
                train_epoch_iou += calculate_iou(pred_masks, true_masks, params.num_classes, device)

                # Calculate the dice loss
                # loss += calculate_dice_loss(pred_masks, true_masks, params.num_classes)    

                # Zero the gradients
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                # Update the training loss
                train_epoch_loss += loss.item()

            val_epoch_iou, val_epoch_loss = validate(model, val_loader, criterion, device, params)

            # visualize an image, its mask, and the predicted mask from the training set and then from the validation set
            visualize_example(train_loader, model, 'train', device, epoch, params)
            visualize_example(val_loader, model, 'validation', device, epoch, params)

            print_str = f"Epoch {epoch + 1}/{params.max_epochs}, Train Loss: {train_epoch_loss / len(train_loader)}, Train IoU: {train_epoch_iou / len(train_loader)}, Val Loss: {val_epoch_loss}, Val IoU: {val_epoch_iou}"

            print(print_str)

            f.write(print_str + "\n")

            if val_epoch_iou > best_val_iou:
                best_val_iou = val_epoch_iou
                torch.save(model, os.path.join(params.checkpoint_dir, 'best_model.pth'))

                print(f"\nBest model saved at epoch {epoch + 1}")
            
            print()

    print("Training complete")

class BlurredTransforms:
    """
    Class for applying transforms.
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
            v2.Resize((80, 120),interpolation=v2.InterpolationMode.BICUBIC,antialias=True), 
            v2.Pad((4,24,4,24)), 
            v2.CenterCrop((80,120)), 
            v2.ToDtype(params.frame_dtype, scale=True), # Convert the image to a PyTorch Tensor and scale pixel values to [0, 1] since thats what the model expects
            v2.Resize((160, 240),interpolation=v2.InterpolationMode.BICUBIC,antialias=True) 
        ])

        mask_transforms = v2.Compose([
            v2.ToDtype(params.mask_dtype), # Convert the mask to a PyTorch Tensor
            v2.Resize(params.resolution, interpolation=InterpolationMode.NEAREST_EXACT), # Resize the mask to the specified resolution
        ])

        image = image_transforms(image)
        mask = mask_transforms(mask)

        return image, mask

def main():
    
    # Load the parameters
    params = Params()

    if params.checkpointing:
        checkpoint_path = params.checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = UNet(in_channels = params.num_frame_channels, out_channels= params.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=params.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Load the data
    data_dir = params.data_dir
    batch_size = params.batch_size
    num_workers = params.num_workers
    train_subset = params.train_subset
    val_subset = params.val_subset
    val_subset = params.val_subset
    pin_memory = params.pin_memory

    train_weird_transforms = BlurredTransforms()
    train_weird_data = VideoSegmentationData(data_dir, 'train', params, train_weird_transforms, train_subset)

    train_transforms = TrainSegmentationTransforms()
    val_transforms = ValSegmentationTransforms()

    train_data = VideoSegmentationData(data_dir, 'train', params, train_transforms, train_subset)
    val_data = VideoSegmentationData(data_dir, 'val', params, val_transforms, val_subset)

    train_dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_data, train_weird_data]), batch_size=batch_size * 2, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Train the model
    train(model, train_dataloader, val_dataloader, criterion, optimizer, device, params)

if __name__ == "__main__":
    main()