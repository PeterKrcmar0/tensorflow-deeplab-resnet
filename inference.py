"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_resnet import *
from pathlib import Path

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 21
SAVE_DIR = './output/'
LEVEL = -1
DATA_PATH = None

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Inference using a DeepLab network.")

    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file.", default=None)
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.", default='./deeplab_resnet.ckpt')
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--level", type=int, default=LEVEL,
                        help="Level of the compression model (1 - 8).")
    parser.add_argument("--data-path", type=str, help="Path to VOC data (optional), can be specified in img_path.", default=DATA_PATH)
    parser.add_argument("--no-gpu", action="store_true", help="Whether to use GPU or not.")
    parser.add_argument("--save-original", action="store_true", help="Whether to save original image (or compressed version).")
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
    
    if args.data_path is not None:
        args.img_path = args.data_path + args.img_path

    # Prepare image.
    og = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)

    # Compress and reconstruct if requested.
    if args.level > 0:
        compressor = get_model_for_level(args.level, latent=False)
        og = tf.cast(og, dtype=tf.uint8)
        og = tf.expand_dims(og, dim=0)
        og = compressor(og)[0]
        og = tf.squeeze(og)
        og.set_shape((None, None, 3))

    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=og)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    
    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
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
    
    msk = decode_labels(preds, num_classes=args.num_classes)
    im = Image.fromarray(msk[0])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    output_file = args.save_dir + Path(args.img_path).stem
    if args.level > 0:
        output_file += f'_mask_anchor2_{args.level}.png'
    else:
        output_file += '_mask_anchor1.png'
    im.save(output_file)

    if args.save_original:
        if args.level > 0:
            if og.dtype.is_floating:
                og = tf.round(og)
            if og.dtype != tf.uint8:
                og = tf.saturate_cast(og, tf.uint8)
        og = sess.run(og)
        og = Image.fromarray(og)
        og_file = args.save_dir + Path(args.img_path).stem
        if args.level > 0:
            og_file += f'_{args.level}.png'
        else:
            og_file += '.jpg'
        og.save(og_file)
    
    print(f'The output file has been saved to {args.save_dir}')

    
if __name__ == '__main__':
    main()
