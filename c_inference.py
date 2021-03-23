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

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 21
SAVE_DIR = './output/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")

    parser.add_argument("img_path", type=str,
                        help="Path to the latent representation npy file.", default='./images/test_indoor2.jpg')

    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.", default='./deeplab_resnet.ckpt')

    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
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
    
    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    
    # Create placeholder for data
    # img_input = tf.placeholder(dtype=tf.uint8, shape=[480,640,3])
    latent_repr = np.load(args.img_path)
    print("////////////////////////////////////////////", latent_repr.shape)
    data = tf.convert_to_tensor(latent_repr)

    # Create network.
    net = DeepLabResNetModel({'data': data}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = [v for v in tf.global_variables() if 'conv1' not in v.name]

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, (1024,1024)) # todo: get size of image from numpy array
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)

    print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", restore_var)
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)
    
    # Perform inference.
    preds = sess.run(pred)
    
    msk = decode_labels(preds, num_classes=args.num_classes)
    im = Image.fromarray(msk[0])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    output_file = args.save_dir + Path(args.img_path).stem + '_mask.png'
    im.save(output_file)
    
    print('The output file has been saved to {}'.format(output_file))

    
if __name__ == '__main__':
    main()
