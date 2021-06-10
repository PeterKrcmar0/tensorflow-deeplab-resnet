"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
from pathlib import Path

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_resnet import *

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 21
SAVE_DIR = './output/'
MODEL = "cResNet"
LEVEL = 1
DATA_PATH = None

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Inference using a compressed-domain network.")

    parser.add_argument("img_path", type=str,
                        help="Path to the image.", default=None)
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.", default='./deeplab_resnet.ckpt')
    parser.add_argument("--save-original", action="store_true",
                        help="If you want to output the original image + mask alongside the predicted mask.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Which model to train.")
    parser.add_argument("--level", type=int, default=LEVEL,
                        help="Level of the compression model (1 - 8).")
    parser.add_argument("--data-path", type=str, help="Path to VOC data.", default=DATA_PATH)
    parser.add_argument("--auto", action="store_true", help="Automatically infer other parameters from model name.")
    parser.add_argument("--no-gpu", action="store_true", help="Don't use the GPU.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    if not args.save_original:
        IMG_PATH = ""
        MASK_PATH = ""
    else:
        MASK_PATH = args.data_path + "SegmentationClassAug/"
        IMG_PATH = args.data_path + "JPEGImages/"

    if args.auto:
        model = Path(args.model_weights).stem
        args.num_classes = 2 if "-bin" in model else 21
        args.level = 1 if "-lvl1" in model else args.level
        model_parts = model.split("-")
        args.model = model_parts[0]
        if "sigma" in model:
            args.model += "-"+model_parts[1]+"-"+model_parts[2]
        print(args.model)
    
    # Prepare image so we can feed it to the compression module.
    image = tf.image.decode_jpeg(tf.io.read_file(IMG_PATH + args.img_path), channels=3)
    image = tf.expand_dims(image, 0)
    image = tf.cast(image, tf.uint8)

    # Get compression model.
    compressor = get_model_for_level(args.level, latent=True, sigma= "sigma" in args.model)

    # Extract latent space
    latent_batch = tf.cast(compressor(image), tf.float32)

    # Create network.
    if args.model == "cResNet":
        net = cResNet91({'data': latent_batch[0]}, num_classes=args.num_classes)
    elif args.model == "cResNet40":
        net = cResNet40({'data': latent_batch[0]}, num_classes=args.num_classes)
    elif args.model == "cResNet42":
        net = cResNet42({'data': latent_batch[0]}, num_classes=args.num_classes)
    elif args.model == "cResNet-sigma-conc":
        net = cResNet_sigma_conc({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, num_classes=args.num_classes)
    elif args.model == "cResNet-sigma-add":
        net = cResNet_sigma_add({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, num_classes=args.num_classes)
    elif args.model == "cResNet-sigma-resblock":
        net = cResNet_sigma_resblock({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, num_classes=args.num_classes)
    else:
        raise Exception(f"Invalid model : {args.model}")

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
    # Set up TF session and initialize variables. 
    if args.no_gpu:
        config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
        config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)
    
    # Perform inference.
    preds = sess.run(pred)

    file_name = Path(args.img_path).stem
    msk = decode_labels(preds, num_classes=args.num_classes)
    im = Image.fromarray(msk[0])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    output_file = args.save_dir + file_name + f'_mask_{args.model}_{args.level}.png'
    im.save(output_file)

    if args.save_original:
        im = Image.open(IMG_PATH + args.img_path)
        im.save(args.save_dir + args.img_path)

        original = tf.image.decode_jpeg(tf.io.read_file(MASK_PATH + file_name + '.png'), channels=1)
        if args.num_classes == 2:
            mask = tf.equal(original, 255)
            original = tf.cast(original, dtype=tf.float32)
            original = tf.clip_by_value(original, 0, 1)
            original = tf.cast(original, dtype=tf.int32)
            original = tf.where_v2(mask, 255, original)
        original = tf.expand_dims(original, 0)
        original = sess.run(original)
        original = decode_labels(original, num_classes=args.num_classes, include=True)
        im = Image.fromarray(original[0])
        im.save(args.save_dir + file_name + '_gt.png')
    
    print('The output file has been saved to {}'.format(output_file))

    
if __name__ == '__main__':
    main()
