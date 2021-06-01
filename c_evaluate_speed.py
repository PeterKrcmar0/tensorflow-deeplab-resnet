"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.lib.shape_base import expand_dims

import tensorflow as tf
import numpy as np

from deeplab_resnet import *

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

DATA_DIRECTORY = './VOC2012/tfci/'
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_DIRECTORY = './output'
MODEL = "cResNet42"
LEVEL = 1

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory where to save miou value.")
    parser.add_argument("--level", type=int, default=LEVEL,
                        help="Level of the compression model (1 - 8).")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model type.")
    parser.add_argument("--num-classes", type=int, default=21, help="Num classes.")
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

    # Create decompressor
    sigma = "-h" in args.model
    decompressor = decompressor_for_level(args.level, latent=True, sigma=sigma)

    # Preprocess image
    #img = tf.placeholder(dtype=tf.float32, shape=(None,None,3))
    #img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    #input_img = tf.concat(axis=2, values=[img_b, img_g, img_r])
    #input_img -= IMG_MEAN
    #input_img = tf.expand_dims(input_img,0)

    y_hat = tf.placeholder(dtype=tf.int32, shape=(None,None,192))
    y_hat = tf.cast(y_hat, tf.float32)
    y_hat = tf.expand_dims(y_hat, 0)
    
    sigma_hat = tf.placeholder(dtype=tf.int32, shape=(None,None,192))
    sigma_hat = tf.cast(sigma_hat, tf.float32)
    sigma_hat = tf.expand_dims(sigma_hat, 0)

    # Create network.
    if args.model == "cResNet":
        net = cResNet_91({'data': y_hat}, num_classes=args.num_classes)
    elif args.model == "cResNet39":
        net = cResNet_39({'data': y_hat}, num_classes=args.num_classes)
    elif args.model == "cResNet42":
        net = cResNet_42({'data': y_hat}, num_classes=args.num_classes)
    elif args.model == "cResNet39-h":
        net = cResNet_39_hyper({'y_hat': y_hat, 'sigma_hat': sigma_hat}, num_classes=args.num_classes)
    elif args.model == "cResNet39-h2":
        net = cResNet_39_hyper2({'y_hat': y_hat, 'sigma_hat': sigma_hat}, num_classes=args.num_classes)
    elif args.model == "cResNet39-h3":
        net = cResNet_39_hyper3({'y_hat': y_hat, 'sigma_hat': sigma_hat}, num_classes=args.num_classes)
    else:
        raise Exception("Invalid model, must be one of (cResNet, cResNet39, cResNet39-h)")
    
    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    dims = tf.shape(raw_output)[1:3,]
    dims = 16 * dims # get dimensions TODO: this is messy and should be extracted from somewhere else...
    raw_output = tf.image.resize_bilinear(raw_output, dims)
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)

    # Iterate over tfci files.
    paths = glob.glob(args.data_dir+"*.tfci")
    paths = sorted(paths)
    print(f"Evaluating {args.model}\'s speed on {len(paths)} images.")

    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)

    times1 = []
    times2 = []
    for i,path in enumerate(paths):
        start = time.time()
        latent = extract_latent_from_file(path, decompressor)
        latent = sess.run(latent)
        mid = time.time()
        if not sigma:
            preds = sess.run(pred, feed_dict={y_hat: latent[0]})
        else:
            preds = sess.run(pred, feed_dict={y_hat: latent[0], sigma_hat: latent[1]})
        
        #preds = decode_labels(preds, num_classes=args.num_classes)
        #plt.imsave(args.save_dir+"/"+Path(path).stem+".png", preds[0])
        end = time.time()
        times1.append(mid - start)
        times2.append(end - mid)
    
    times1 = np.array(times1)
    times2 = np.array(times2)
    times = times1 + times2
    
    print(times1.mean(), "+-", times1.std())
    print(times2.mean(), "+-", times2.std())
    print(times.mean(), "+-", times.std())

    with open(f'{args.save_dir}/times.txt', 'a') as f:
        f.write(f'{args.restore_from.split("/")[-1]} {times.mean()} {times.std()}\n')

if __name__ == '__main__':
    main()
