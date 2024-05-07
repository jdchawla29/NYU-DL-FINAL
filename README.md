# NYU-DL-FINAL

## Task

The task on hidden set is to use the first 11 frames to generate the semantic
segmentation mask of the last frame (the 22nd frame).

## Dataset

The dataset consists of 13000 video clips of 22 frames each in the `unlabeled` folder. The `val` and `train` folder are for inference.

available here: https://drive.google.com/file/d/1iYTFuf4DgxgYQzTQ_2da1vC9es_niPRr/view?usp=drive_link

Folder Structure \
  &nbsp;&nbsp;&nbsp;&nbsp; unlabeled/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; video_02000/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; image_0.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; image_1.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; image_21.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; video_02001/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; video_... \
&nbsp;&nbsp;&nbsp;&nbsp; train/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ...\
&nbsp;&nbsp;&nbsp;&nbsp; val/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ...

Note that there are also masks.npy files which are meant for segmentation.

## Models

We use a two-stage pipeline for next-frame prediction and semantic segmentation.

### Next-frame prediction

We use stdiff model for next-frame prediction. The model is trained on the unlabeled dataset. The inference script predicts the next 11 frames given the first 11 frames on the val/hidden dataset. 

### Semantic segmentation

We use a simple U-net model for semantic segmentation. The model is trained on the train dataset and evaluated on the resulting frames from stdiff.

For training and inference details, head to over to the directories `1. stdiff` and `2. segmentation`.