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

import tensorflow as tf
import numpy as np

from deeplab_resnet import * #cResNetModel, cResNet_39, ImageReader, prepare_label, get_model_for_level

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
BIN_CLASSES = [
    'background', 'foreground'
]

DATA_DIRECTORY = './VOC2012'
DATA_LIST_PATH = './dataset/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = -1
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_DIRECTORY = './output'
LEVEL = 1
MODEL = "cResNet39"

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory where to save miou value.")
    parser.add_argument("--level", type=int, default=LEVEL,
                        help="Level of the compression model (1 - 8).")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Which model to train (cResNet, cResNet39, resNet).")
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

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Create compression model.
    compressor = get_model_for_level(args.level, latent="cResNet" in args.model, sigma="-h" in args.model)

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord, 
            latent=True,
            binary=args.num_classes == 2)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Extract latent space
    latent_batch = tf.cast(compressor(image_batch), tf.float32)

    # Create network.
    if args.model == "cResNet":
        net = cResNet_91({'data': latent_batch[0]}, num_classes=args.num_classes)
    elif args.model == "cResNet39":
        net = cResNet_39({'data': latent_batch[0]}, num_classes=args.num_classes)
    elif args.model == "cResNet42":
        net = cResNet_42({'data': latent_batch[0]}, num_classes=args.num_classes)
    elif args.model == "cResNet39-h":
        net = cResNet_39_hyper({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, num_classes=args.num_classes)
    elif args.model == "cResNet39-h2":
        net = cResNet_39_hyper2({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, num_classes=args.num_classes)
    elif args.model == "cResNet39-h3":
        net = cResNet_39_hyper3({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, num_classes=args.num_classes)
    else:
        raise Exception("Invalid model, must be one of (cResNet, cResNet39, cResNet39-h)")

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.

    # mIoU
    pred = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(gt, args.num_classes - 1)), 1)  # ignore all labels >= num_classes
    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    pred = tf.gather(pred, indices)
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes)

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

    # Get confusion matrix tensor.
    confusion_matrix = tf.get_default_graph().get_tensor_by_name('mean_iou/total_confusion_matrix:0')

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    if args.num_steps == -1:
        with open(args.data_list, 'r') as f:
            args.num_steps = len(f.readlines())
    print(f'Running evalutation on {args.num_steps} images.')
    for step in range(args.num_steps):
        preds, _ = sess.run([pred, update_op])
        if step % 100 == 0:
            print('step {:d}'.format(step))
    miou_val = mIoU.eval(session=sess)
    print('Mean IoU: {:.3f}'.format(miou_val))
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    with open(f'{args.save_dir}/miou.txt', 'a') as f:
        f.write(f'{args.restore_from} {miou_val}\n')

    # Get the per-class IoUs as well from the confusion matrix.
    c = confusion_matrix.eval(session=sess)
    TP = np.diag(c)
    FP = c.sum(axis=0) - np.diag(c)
    FN = c.sum(axis=1) - np.diag(c)
    IOU = TP / (TP + FP + FN)
    np.save(f'{args.save_dir}/confusion_matrix_{args.restore_from.split("/")[-1]}.npy', c)
    our_miou = np.nanmean(IOU)
    # get pixel accuracy as well if we are doing binary classification
    if (args.num_classes == 2):
        TN = c[0,0] # both agree on background
        TP = c[1,1] # both agree on object
        FN = c[1,0] # predicts background but was object
        FP = c[0,1] # predicts object but was background
        (TN, FP, FN, TP) = c.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + TN)
        jaccard = TP / (TP + FP + FN)
        print(accuracy, jaccard)
        with open(f'{args.save_dir}/pixel_acc.txt', 'a') as f:
            f.write(f'{args.restore_from} {accuracy} {jaccard}\n')
    print('Our mean IoU: {:.3f}'.format(our_miou))
    for i,(v,c) in enumerate(zip(list(IOU),VOC_CLASSES)):
        print(f'IoU for class {i}: {v:.3f} ({c})')

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
