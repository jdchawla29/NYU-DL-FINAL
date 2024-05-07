# Training and Inference using UNET

To train the UNet model, simply run 

```
python train.py
```

Make sure to take a look at the utils/params.py file to see what the hyperparameters are. 

This will create a folder in the checkpoints folder (which will be created if it does not exist in segmentation directory). This folder will be named based on the time that you ran the above function (i.e. something like 2024-04-14_04_15_03_285085)

To perform inference on the hidden set (without predicting the 22nd frame, simply predicting the 11th frame's mask) using this UNet model, run the train.py file first, then find the path for "best_model.pth" in the "checkpoints" folder which is located as mentioned above.

To run the inference using only 11th frame on hidden set, simply run

```
python hidden_set_inference.py --model_path [/path/to/best__segmentation_model.pth]
```

and make sure you have your hidden data located under NYU-DL-FINAL/data/hidden

To run the inference using reconstructed 22nd frames, simply run

```
python hidden_set_inference.py --model_path [/path/to/best__segmentation_model.pth] --reconstructed_data_dir [/path/to/reconstructed/22nd/frames]
```

This will add a folder called hidden_set_masks to the checkpoints folder where you got the best_model.pth from, which will contain 5 example images (the image + the corresponding predicted mask). It will also have a file called all_hidden_set_masks.tensor which contains all 5000 predicted segmentation masks.

You will need a different conda environment for the segmentation model and the reconstruction model. Run the following commands to create a conda environment for training and testing the UNet model.

```
conda create -n segmentation python=3.11
conda activate seg
conda install matplotlib tqdm pytorch torchvision torchaudio torchmetrics jupyter pytorch-cuda=11.8 -c pytorch -c nvidia -y
```