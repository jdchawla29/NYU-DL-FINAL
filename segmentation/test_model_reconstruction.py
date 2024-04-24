# general imports
import os
import numpy as np
from tqdm import tqdm
import argparse

# torch imports
import torch
import torch.nn.functional as F

# local imports
from utils.params import Params
from utils.visualize import visualize_predicted_example
from utils.data_loaders import VideoSegmentationData, get_hidden_set_loader
from utils.data_transforms import ValSegmentationTransforms
from torchvision.transforms import v2, InterpolationMode
from torchmetrics import JaccardIndex as IoU
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

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
def validate(model, val_loader, predicted_val_loader, mask_criterion, image_criterion, device, params = Params()):

    print("\nValidating...\n")

    model.eval()

    total_image_loss = 0
    total_original_loss = 0
    total_original_iou = 0

    total_predicted_loss = 0
    total_predicted_iou = 0

    i = 0

    for ((original_images, true_masks), (predicted_images)) in tqdm(zip(val_loader, predicted_val_loader), total=len(predicted_val_loader), desc="Processing batches"):

        # ground truth masks
        true_masks = true_masks.squeeze(1).to(device)

        ############################ Original Data ############################
        original_images = original_images.to(device)

        original_pred_masks = model(original_images)
        original_loss = mask_criterion(original_pred_masks, true_masks)

        original_iou = calculate_iou(original_pred_masks, true_masks, params.num_classes, device)
        
        total_original_iou += original_iou
        total_original_loss += original_loss.item()

        mask_original_pred_argmax = torch.argmax(F.softmax(original_pred_masks, dim=1), dim=1).squeeze(0)

        visualize_predicted_example(original_images[0].detach().cpu(), true_masks.squeeze(1)[0].detach().cpu(), mask_original_pred_argmax.detach().cpu(), f"True vs Predicted Mask (Original Val Set) - IOU ({original_iou})", os.path.join(params.out_dir, f"original_video_{i}_image_22_output.png"), params)

        ############################ Predicted Data ############################

        predicted_images = predicted_images.to(device)

        predicted_pred_masks = model(predicted_images)
        predicted_loss = mask_criterion(predicted_pred_masks, true_masks)

        predicted_iou = calculate_iou(predicted_pred_masks, true_masks, params.num_classes, device)

        total_predicted_iou += predicted_iou
        total_predicted_loss += predicted_loss.item()

        mask_predicted_pred_argmax = torch.argmax(F.softmax(predicted_pred_masks, dim=1), dim=1).squeeze(0)

        visualize_predicted_example(predicted_images[0].detach().cpu(), true_masks.squeeze(1)[0].detach().cpu(), mask_predicted_pred_argmax.detach().cpu(), f"True vs Predicted Mask (Predicted Val Set) - IOU ({predicted_iou})", os.path.join(params.out_dir, f"predicted_video_{i}_image_22_output.png"), params)

        ############################ COMPARE ORIGINAL AND PREDICTED IMAGES ############################

        # compute image loss
        total_image_loss += image_criterion(target = original_images, preds = predicted_images)

        i += 1

    model.train()

    print()

    print(f"Original Data: Average IOU: {total_original_iou / len(predicted_val_loader)}, Average Loss: {total_original_loss / len(predicted_val_loader)}")
    print(f"Predicted Data: Average IOU: {total_predicted_iou / len(predicted_val_loader)}, Average Loss: {total_predicted_loss / len(predicted_val_loader)}")

    print(f"Average Image Loss: {total_image_loss / len(predicted_val_loader)}")

    print("Note that the image loss is calculated using the SSIM metric, which is a measure of structural similarity between two images. The closer the value is to 1, the more similar the images are.")
    print("According to the SSIM documentation, a value of 0.98 or higher indicates that the images are almost identical.")

def main():
    parser = argparse.ArgumentParser(description='Comparing the stdiff 22nd frame reconstruction with the original 22nd frame and seeing the impact on the mask prediction')
    parser.add_argument('--model_path', help = 'Path to the saved torch UNet model to use for inference', required = True, type = str)
    parser.add_argument('--reconstructed_img_dir', help = 'Path to the reconstructed data directory', required = False, type = str, default=None)
    parser.add_argument('--out_dir', help = 'Path to the output directory to store example images', required = True, type = str)
    
    args = parser.parse_args()

    model_path = args.model_path
    out_dir = args.out_dir
    reconstructed_img_dir = args.reconstructed_img_dir

    # Load the parameters
    params = Params()

    params.out_dir = out_dir

    # try to create the output directory and if it already exists, ignore the error
    os.makedirs(params.out_dir, exist_ok=True)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the unet model
    UNet = torch.load(model_path)

    # Load the data
    data_dir = params.data_dir
    num_workers = params.num_workers
    val_subset = params.val_subset
    pin_memory = params.pin_memory

    val_transforms = ValSegmentationTransforms()

    val_data = VideoSegmentationData(data_dir, 'val', params, val_transforms, val_subset, get_last_frame_only=True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    mask_criterion = torch.nn.CrossEntropyLoss()
    image_criterion = SSIM().to(device)

    params.reconstructed_img_dir = reconstructed_img_dir

    # Load the data (although its called hidden_set_loader, here we are using it to load the predicted val set - normally we only use this for the hidden set)
    predicted_val_loader = get_hidden_set_loader(params)

    # Test the model on both the original and reconstructed data
    validate(UNet, val_loader, predicted_val_loader, mask_criterion, image_criterion, device, params)


if __name__ == "__main__":
    main()