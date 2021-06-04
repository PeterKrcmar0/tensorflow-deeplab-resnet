# Semantic segmentation in the latent space

This repo includes code for performing semantic segmentation directly on the latent representation of an image compressed by a learning-based compression model, in this case the "Variational Image Compression with a Scale Hyperprior" by Ballé et al. [1](https://arxiv.org/abs/1802.01436).

It is based on [this](https://github.com/DrSleep/tensorflow-deeplab-resnet) DeepLabv2 (re-)implementation.

## Model Description

## Requirements

The code requires Tensorflow 1.15 and tensorflow-compression 1.3, and was tested using python 3.6 and 3.7.

To install the required python packages (without Tensorflow) run
```bash
pip install tensorflow_compression==1.3
pip install tf-slim
pip install -r requirements.txt
```

## Additional downloads

**Model weights**, pretrained on MS-COCO and converted from the original `.caffemodel` model can be downloaded [here](https://drive.google.com/open?id=0B_rootXHuswsZ0E4Mjh1ZU5xZVU).

Augmented PASCAL VOC 2012 **dataset** can be downloaded [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0).

## Scripts

Scripts that are prefixed with **c_** use networks that operate on the latent space instead of RGB images.

All scripts can be called with the `--help` option to list all options.
As an example, assuming you have downloaded and extracted the files in the "Additional downloads" section to the root folder, training a basic compressed-domain network with the default training parameters, can be done with:
```bash
python c_train.py --model cResNet42 --level 1
```

And for inference, you can use:
```bash
python c_inference.py <path/to/image> <path/to/ckpt> --model cResNet42 --level 1
```