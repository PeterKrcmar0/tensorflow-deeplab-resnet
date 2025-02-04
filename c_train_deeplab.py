"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
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

BATCH_SIZE = 10
DATA_DIRECTORY = './VOC2012'
DATA_LIST_PATH = './dataset/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '320,320'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 20001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = None # './deeplab_resnet_init.ckpt'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
ALPHA = 0.001
LEVEL = 1
MODEL = "cResNet"
LEARNING = "poly"

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a compressed-domain network using the setup used in deeplab. For better results, consider using c_train.py")
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
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
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
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--level", type=int, default=LEVEL,
                        help="Level of the compression model (1 - 8).")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Which model to train.")
    parser.add_argument("--lr-policy", type=str, default=LEARNING,
                        help="How to adapt learning rate (poly, exp)")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Alpha for exp learning rate policy")
    parser.add_argument("--update-bn", action="store_true",
                        help="If gamma and beta should be learned for the first layers")
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
      saver: TensorFlow Saver object.
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
    compressor = get_model_for_level(args.level, latent=True, sigma= "sigma" in args.model)
    
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
        net = cResNet91({'data': latent_batch[0]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    elif args.model == "cResNet40":
        net = cResNet40({'data': latent_batch[0]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    elif args.model == "cResNet42":
        net = cResNet42({'data': latent_batch[0]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    elif args.model == "cResNet-sigma-conc":
        net = cResNet_sigma_conc({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    elif args.model == "cResNet-sigma-add":
        net = cResNet_sigma_add({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    elif args.model == "cResNet-sigma-resblock":
        net = cResNet_sigma_resblock({'y_hat': latent_batch[0], 'sigma_hat': latent_batch[1]}, is_training=args.is_training, is_training2=args.is_training, num_classes=args.num_classes)
    else:
        raise Exception("Invalid model")
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
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    restore_var = [v for v in restore_var if 'correct_channels' not in v.name]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    beta_gamma_trainable = [v for v in tf.trainable_variables() if 'correct_channels' in v.name and ('beta' in v.name or 'gamma' in v.name)] # update beta and gamma of correct_channels layers
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
    
    # Predictions: reshape logits and downscale GT
    if not args.upscale:
        raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
        label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
        raw_gt = tf.reshape(label_proc, [-1,])
        
    # Upscale logits instead of downscaling GT
    else:
        raw_prediction = tf.image.resize_bilinear(raw_output, input_size) # upscale
        raw_prediction = tf.reshape(raw_prediction, [-1, args.num_classes]) # reshape
        raw_gt = tf.reshape(label_batch, [-1,])

    # Mask out values larger than number of classes (ignore label)
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    
    # Loss summary.
    pixel_loss_summary = tf.summary.scalar('pixel_loss', tf.reduce_mean(loss))
    total_loss_summary = tf.summary.scalar('total_loss', reduced_loss)
    loss_summary = tf.summary.merge([pixel_loss_summary, total_loss_summary])

    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
    # Image summary (currently disabled due to memory)
    images_summary = image_batch[:args.save_num_images] # original images, should be uint8   #tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)
    
    total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
    
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())
   
    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.lr)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    if args.lr_policy == "poly":
        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    elif args.lr_policy == "exp":
        learning_rate = tf.scalar_mul(base_lr, tf.pow(args.alpha, (step_ph / args.num_steps)))
    opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    if args.update_bn:
        conv_trainable += beta_gamma_trainable

    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)) : ]
    
    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
    
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

    model_name = f'{args.model}-lvl{args.level}'
    if args.num_classes == 2:
        model_name += '-bin'

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }
        
        loss_value, loss_sum,  _ = sess.run([reduced_loss, loss_summary, train_op], feed_dict=feed_dict)
        summary_writer.add_summary(loss_sum, step)
        
        if step % args.save_pred_every == 0:
            #loss_value, images, labels, preds, summary, loss_sum, _ = sess.run([reduced_loss, image_batch, label_batch, pred, total_summary, loss_summary, train_op], feed_dict=feed_dict)
            #summary_writer.add_summary(loss_sum, step)
            #summary_writer.add_summary(summary, step)
            save(saver, sess, args.snapshot_dir, step, model_name)
            
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
