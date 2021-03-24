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

from deeplab_resnet import cResNetModel, ImageReader, decode_labels, prepare_label, get_latent_space, get_model_for_level

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
    
    # Prepare image so we can feed it to the compression module.
    image = tf.image.decode_jpeg(tf.io.read_file(args.img_path), channels=3)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, (448,448))
    image = tf.cast(image, tf.uint8)

    # Get compression model.
    c_model = get_model_for_level(5)

    # Extract latent space
    latent_repr = c_model(image)[0]
    latent_repr = tf.cast(latent_repr, dtype=tf.float32)
    print("-------------", latent_repr.shape)

    # Create network.
    net = cResNetModel({'data': latent_repr}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    not_restore = [v for v in tf.global_variables() if 'correct_channels' in v.name] # original model doesn't contain weights for 'correct_channels' (TODO also remove the weights from next layers ???)
    print(not_restore)
    restore_var = [v for v in tf.global_variables() if v not in not_restore]

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
    # Set up TF session and initialize variables. 
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

    # tf.train.write_graph(sess.graph, ".", "test.pb", as_text=False)
    
    msk = decode_labels(preds, num_classes=args.num_classes)
    im = Image.fromarray(msk[0])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    output_file = args.save_dir + Path(args.img_path).stem + '_mask.png'
    im.save(output_file)
    
    print('The output file has been saved to {}'.format(output_file))

    
if __name__ == '__main__':
    main()
