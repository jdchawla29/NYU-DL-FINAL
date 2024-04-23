import matplotlib.pyplot as plt

# torch imports
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import v2, InterpolationMode

# local imports
from utils.params import Params

def visualize_predicted_example(image, true_mask, predicted_mask, title, save_path, params = Params()):
    """
    Visualize an image, (optionally) its true mask, and the predicted mask.
    
    Args:
        image (torch.Tensor): The input image.
        true_mask (torch.Tensor): The true mask.
        predicted_mask (torch.Tensor): The predicted mask.
    """

    # print('Image shape:', image.shape)
    # print('Mask shape:', true_mask.shape)
    # print('Predicted mask shape:', predicted_mask.shape)

    # Convert the torch.tensor image to PIL for easy visualization
    image = to_pil_image(image)

    if image.size != (3, 160, 240):
        image = v2.Resize((160, 240))(image)

    # squeeze the mask to remove the channel dimension so that it can be visualized
    if true_mask is not None and true_mask.shape[0] == 1:
        true_mask = true_mask.squeeze(0)

        if true_mask.shape != (160, 240):
            true_mask = v2.Resize((160, 240), interpolation=InterpolationMode.NEAREST)(true_mask)

    if predicted_mask.shape[0] == 1:
        predicted_mask = predicted_mask.squeeze(0)

        if predicted_mask.shape != (160, 240):
            predicted_mask = v2.Resize((160, 240), interpolation=InterpolationMode.NEAREST)(predicted_mask)

    num_plots = 3 if true_mask is not None else 2
    i = 0

    # Plotting
    _, ax = plt.subplots(1, num_plots, figsize=(12, 6))

    # set a title for the overall plot
    plt.suptitle(title, fontsize=16)

    ax[i].imshow(image)
    ax[i].set_title('Image')
    ax[i].axis('off')
    i += 1
    
    if true_mask is not None:
        ax[i].imshow(true_mask)
        ax[i].set_title('True Mask')
        ax[i].axis('off')
        i += 1

    ax[i].imshow(predicted_mask)
    ax[i].set_title('Predicted Mask')
    ax[i].axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', dpi = params.dpi)
    plt.close()