# general imports
import os
from tqdm import tqdm
import argparse

# torch imports
import torch
import torch.nn.functional as F

# local imports
from utils.params import Params
from utils.visualize import visualize_predicted_example
from utils.data_loaders import get_hidden_set_loader
from torchvision.transforms import v2, InterpolationMode

@torch.inference_mode()
def model_inference(model, data_loader, device, params = Params()):

    model.eval()

    masks = []

    for i, image in enumerate(tqdm(data_loader, desc='Hidden Set Inference')):

        image = image.to(device)

        predicted_masks = model(image)

        mask_pred_argmax = torch.argmax(F.softmax(predicted_masks, dim=1), dim=1) # [1, params.resolution[0], params.resolution[1]

        if mask_pred_argmax.shape[1] != 160 or mask_pred_argmax.shape[2] != 240:
            # resize the mask to the original resolution of 160x240
            mask_pred_argmax = v2.Resize((160, 240), interpolation=InterpolationMode.BICUBIC)(mask_pred_argmax)

        if i % 1000 == 0: # visualize every 1000th mask
            visualize_predicted_example(image[0].detach().cpu(), None, mask_pred_argmax.detach().cpu(), f"Predicted Mask for Hidden Set Video {i}", os.path.join(params.hidden_set_mask_dir, f"hidden_{i}.png"), params)

        masks.append(mask_pred_argmax.squeeze(0).detach().cpu())

    masks = torch.stack(masks)

    assert masks.shape == (5000, 160, 240), f"Expected masks to have shape (5000, 160, 240), but got {masks.shape}"

    # save the masks to a file
    torch.save(masks, os.path.join(params.hidden_set_mask_dir, 'all_hidden_set_masks.tensor'))

    return

def main():
    parser = argparse.ArgumentParser(description='Running segmentation mask inference on the hidden set using only the 11th frame')
    parser.add_argument('--model_path', help = 'Path to the saved torch model to use for inference', required = True, type = str)
    parser.add_argument('--reconstructed_img_dir', help = 'Path to the reconstructed data directory', required = False, type = str, default=None)
    
    args = parser.parse_args()

    model_path = args.model_path
    reconstructed_img_dir = args.reconstructed_img_dir

    # Load the parameters
    params = Params()

    params.hidden_set_mask_dir = os.path.join(os.path.dirname(model_path), 'hidden_set_masks')

    if not os.path.exists(params.hidden_set_mask_dir):
        os.makedirs(params.hidden_set_mask_dir)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the unet model
    inference_model = torch.load(model_path)

    params.reconstructed_img_dir = reconstructed_img_dir

    # Load the data
    hidden_set_loader = get_hidden_set_loader(params)

    # Test the model on the hidden set
    model_inference(inference_model, hidden_set_loader, device, params)


if __name__ == "__main__":
    main()