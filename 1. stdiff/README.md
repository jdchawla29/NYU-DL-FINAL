

# STDiff</h1>

## Installation
1. Install the custom diffusers library
```bash
git clone https://github.com/XiYe20/CustomDiffusers.git
cd CustomDiffusers
pip install -e .
```
2. Install the requirements
```bash
pip install -r requirements.txt
```

## Training and Evaluation
Accelerate is used for training. The configuration files are placed inside stdiff/configs.

### Training
1. Check train.sh, modify the visible gpus, num_process, modify the config.yaml file
2. Training
```bash
. ./train.sh
```

### Test
1. Check inference.sh, modify config.yaml for inference
2. Test
```bash
. ./inference.sh
```

## Results
The results are stored in the output folder. The results are stored in the form of images frames generated corresponding each video in the val/hidden dataset

## Acknowledgements
This codebase is adapted from adapted from [XiYe20/STDiffProject](https://github.com/XiYe20/STDiffProject) (paper: [arXiv link](https://arxiv.org/abs/2312.06486))