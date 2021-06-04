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

DATA_DIRECTORY = './VOC2012/tfci/'
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_DIRECTORY = './output'
MODEL = "cResNet42"
LEVEL = 1
NUM_ITERS = -1

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate speed of comrpessed-domain network.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the data (images if --rt or tfci files otherwise)")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory where to save speed values. (debug)")
    parser.add_argument("--level", type=int, default=LEVEL,
                        help="Level of the compression model (1 - 8).")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model type.")
    parser.add_argument("--num-classes", type=int, default=21, help="Number of classes.")
    parser.add_argument("--rt", action="store_true", help="If using realtime pipeline (rgb images, not tfci)")
    parser.add_argument("--no-gpu", action="store_true", help="To turn of gpu.")
    parser.add_argument("--num-iter", type=int, default=NUM_ITERS, help="Number of images to evaluate.")
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

    # Get codec
    sigma = "sigma" in args.model

    if args.rt:
        codec = get_model_for_level(args.level, latent=True, sigma=sigma)
    else:
        codec = decompressor_for_level(args.level, latent=True, sigma=sigma)

    # Prepare input
    n_channels = 300 if args.level > 5 else 192
    y_hat = tf.placeholder(dtype=tf.int32, shape=(None,None,n_channels))
    y_hat = tf.cast(y_hat, tf.float32)
    y_hat = tf.expand_dims(y_hat, 0)
    
    sigma_hat = tf.placeholder(dtype=tf.int32, shape=(None,None,n_channels))
    sigma_hat = tf.cast(sigma_hat, tf.float32)
    sigma_hat = tf.expand_dims(sigma_hat, 0)

    # Create network.
    if args.model == "cResNet":
        net = cResNet91({'data': y_hat}, num_classes=args.num_classes)
    elif args.model == "cResNet40":
        net = cResNet40({'data': y_hat}, num_classes=args.num_classes)
    elif args.model == "cResNet42":
        net = cResNet42({'data': y_hat}, num_classes=args.num_classes)
    elif args.model == "cResNet-sigma-conc":
        net = cResNet_sigma_conc({'y_hat': y_hat, 'sigma_hat': sigma_hat}, num_classes=args.num_classes)
    elif args.model == "cResNet-sigma-add":
        net = cResNet_sigma_add({'y_hat': y_hat, 'sigma_hat': sigma_hat}, num_classes=args.num_classes)
    elif args.model == "cResNet-sigma-resblock":
        net = cResNet_sigma_resblock({'y_hat': y_hat, 'sigma_hat': sigma_hat}, num_classes=args.num_classes)
    else:
        raise Exception("Invalid model")
    
    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    dims = tf.shape(raw_output)[1:3,]
    dims = 16 * dims # get dimensions TODO: this is not the correct way to do this
    raw_output = tf.image.resize_bilinear(raw_output, dims)
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.

    # Set up tf session and initialize variables.
    if args.no_gpu:
        config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
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

    # Iterate over tfci files or images
    if args.rt:
        paths = glob.glob(args.data_dir+"*.jpg")
    else:
        paths = glob.glob(args.data_dir+"*.tfci")
    paths = sorted(paths)
    if args.num_iter > 0:
        paths = paths[:min(args.num_iter,len(paths))]
    print(f"Evaluating {args.model}\'s speed on {len(paths)} images.")

    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)

    times1 = []
    times2 = []
    for i,path in enumerate(paths):
        if i % 10 == 0:
            print(f"{i}/{len(paths)}")

        # realtime mode: read image -> encode to y (and sigma) -> inference
        if args.rt:
            # read image (not timed)
            img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
            img = tf.expand_dims(img, 0)
            img = tf.cast(img, tf.uint8)
            img = sess.run(img)

            # encode to y (and sigma)
            t1 = time.time()
            img = tf.constant(img)
            latent = sess.run(codec(img))
            
            # inference
            t2 = time.time()
            if not sigma:
                preds = sess.run(pred, feed_dict={y_hat: latent[0]})
            else:
                preds = sess.run(pred, feed_dict={y_hat: latent[0], sigma_hat: latent[1]})

        # not realtime: read tfci file -> extract y (and sigma) -> inference
        else:
            # read tfci file (not timed)
            with tf.io.gfile.GFile(path, "rb") as f:
                bitstring = f.read()

            # extract y (and sigma)
            t1 = time.time()
            latent = extract_latent_from_bitstring(bitstring, codec)
            latent = sess.run(latent)

            # inference
            t2 = time.time()
            if not sigma:
                preds = sess.run(pred, feed_dict={y_hat: latent[0]})
            else:
                preds = sess.run(pred, feed_dict={y_hat: latent[0], sigma_hat: latent[1]})

        t3 = time.time()

        # DEBUG: save predicted mask
        # preds = decode_labels(preds, num_classes=args.num_classes)
        # plt.imsave(args.save_dir+"/"+Path(path).stem+".png", preds[0])
        
        times1.append(t2 - t1)
        times2.append(t3 - t2)
    
    times1 = np.array(times1)
    times2 = np.array(times2)
    times = times1 + times2
    
    print(f"Mean codec time: {times1.mean():.3f}")
    print(f"Mean inference time: {times2.mean():.3f}")
    print(f"Mean total time: {times.mean():.3f}")

    # DEBUG: store data in file
    #with open(f'{args.save_dir}/times.txt', 'a') as f:
    #    string = f"{times1.mean():.3f} $\\pm$ {times1.std():.3f} & {times2.mean():.3f} $\\pm$ {times2.std():.3f} & {times.mean():.3f} $\\pm$ {times.std():.3f}"
    #    f.write(f'{args.restore_from.split("/")[-1]} {args.rt} {string}\n')

if __name__ == '__main__':
    main()
