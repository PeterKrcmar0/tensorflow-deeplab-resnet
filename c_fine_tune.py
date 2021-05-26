"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script fine-tunes the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
Only the last 'fc1_voc12' layers are being trained.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from deeplab_resnet import *

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 4
DATA_DIRECTORY = './VOC2012'
DATA_LIST_PATH = './dataset/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '320,320'
LEARNING_RATE = 1e-4
NUM_CLASSES = 21
NUM_STEPS = 20001
RANDOM_SEED = 1234
RESTORE_FROM = './deeplab_resnet_init.ckpt'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = './snapshots_finetune/'
LEVEL = 1
MODEL = "cResNet"

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--level", type=int, default=LEVEL,
                        help="Level of the compression model (1 - 8).")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Which model to train (cResNet, cResNet39, resNet).")
    parser.add_argument("--upscale", action="store_true",
                        help="If the prediction should be upscaled to compute the loss.")
    return parser.parse_args()

def save(saver, sess, logdir, step, model_name):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = model_name+'.ckpt' #'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=step==0)
   print('The checkpoint has been created.')

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
    """Create the model and start the training."""
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    tf.set_random_seed(args.random_seed)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Create compression model.
    compressor = get_model_for_level(args.level, latent="cResNet" in args.model, include_hyperprior= "-h" in args.model)
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord,
            latent=True,
            binary=args.num_classes == 2)
        image_batch, label_batch = reader.dequeue(args.batch_size)

    latent_batch = tf.cast(compressor(image_batch), tf.float32)
    
    # Create network.
    if args.model == "cResNet":
        net = cResNet_91({'data': latent_batch[0]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    elif args.model == "cResNet39":
        net = cResNet_39({'data': latent_batch[0]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    elif args.model == "cResNet42":
        net = cResNet_42({'data': latent_batch[0]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    elif args.model == "cResNet39-h":
        net = cResNet_39_hyper({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    elif args.model == "cResNet39-h2":
        net = cResNet_39_hyper2({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    elif args.model == "cResNet39-h3":
        net = cResNet_39_hyper3({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    else:
        raise Exception("Invalid model, must be one of (cResNet, cResNet39, cResNet39-h)")
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    # Restore all variables, or all except the last ones.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    trainable = [v for v in tf.trainable_variables() if 'fc1_voc12' in v.name] # Fine-tune only the last layers
    
    if args.upscale:
        raw_prediction = tf.image.resize_bilinear(raw_output, input_size) # upscale
        raw_prediction = tf.reshape(raw_prediction, [-1, args.num_classes]) # reshape
        raw_gt = tf.reshape(label_batch, [-1,])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1) # mask out 255
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        prediction = tf.gather(raw_prediction, indices)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    else:
        prediction = tf.reshape(raw_output, [-1, args.num_classes])
        label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes)
        gt = tf.reshape(label_proc, [-1, args.num_classes])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    
    # Pixel-wise softmax loss.
    reduced_loss = tf.reduce_mean(loss)
    
    # Processed predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, input_size)
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
    # Image summary.
    if args.save_num_images > args.batch_size:
        args.save_num_images = args.batch_size
        print(f'Number of images to save was larger than batch size, setting to {args.save_num_images}.')
    images_summary = image_batch[:args.save_num_images] #tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)
    
    total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
    loss_summary = tf.summary.scalar('loss', reduced_loss)
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())
   
    # Define loss and optimisation parameters.
    optimiser = tf.train.AdamOptimizer(learning_rate=args.lr)
    optim = optimiser.minimize(reduced_loss, var_list=trainable)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=40)
    
    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        
    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        
        if step % args.save_pred_every == 0:
            loss_value, images, labels, preds, summary, summary2, _ = sess.run([reduced_loss, image_batch, label_batch, pred, total_summary, loss_summary, optim])
            summary_writer.add_summary(summary, step)
            summary_writer.add_summary(summary2, step)
            save(saver, sess, args.snapshot_dir, step)
        else:
            loss_value, summary2, _ = sess.run([reduced_loss, loss_summary, optim])
            summary_writer.add_summary(summary2, step)
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
