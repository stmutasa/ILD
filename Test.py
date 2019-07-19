""" Training the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import os
import time

import ILDModel as network
import SODTester as SDT
import SODLoader as SDL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define an instance of the loader and testing file
sdl = SDL.SODLoader(os.getcwd())
sdt = SDT.SODTester(False, False)

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('data_dir', 'data/', """Path to the data directory.""")
tf.app.flags.DEFINE_string('testing_dir', 'testing/', """Path to the testing directory.""")
tf.app.flags.DEFINE_string('test_files', '1', """Testing files""")
tf.app.flags.DEFINE_integer('box_dims', 256, """dimensions to save files""")
tf.app.flags.DEFINE_integer('network_dims', 128, """dimensions for the network input""")
tf.app.flags.DEFINE_float('noise_threshold', 10, 'Amount of Gaussian noise to apply')

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('epoch_size', 2135, """How many examples""")
tf.app.flags.DEFINE_integer('batch_size', 8, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 1.0, """ Keep probability""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")
tf.app.flags.DEFINE_float('dice_threshold', 0.1, """ The threshold value to declare PE""")
tf.app.flags.DEFINE_float('size_threshold', 1.0, """ The size threshold value to declare detected PE""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-5, """ The gamma value for regularization loss""")

# Define a custom training class
def test():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default():

        # Load the images and labels.
        _, validation = network.inputs(skip=True)

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Perform the forward pass:
        if FLAGS.network_dims == 128: logits, l2loss = network.forward_pass_dense(validation['image'], phase_train=phase_train)
        else: logits, l2loss = network.forward_pass_256(validation['image'], phase_train=phase_train)

        # Retreive the softmax for testing purposes
        softmax = tf.nn.softmax(logits)

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        while True:

            # config Proto sets options for configuring the session like run on GPU, allocate GPU memory etc.
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as mon_sess:

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state('training/')

                # Initialize the variables
                mon_sess.run(var_init)

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the learned variables
                    restorer = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

                    # Restore the graph
                    restorer.restore(mon_sess, ckpt.model_checkpoint_path)

                    # Extract the epoch
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

                # Initialize the thread coordinator
                coord = tf.train.Coordinator()

                # Start the queue runners
                threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)

                # Initialize the step counter
                tot, TP, TN, FP, FN, DICE, total, step = 0, 0, 0, 0, 0, 0, 1e-8, 0

                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                try:
                    while step < max_steps:

                        # Retreive the predictions and labels
                        preds, labs, egs = mon_sess.run([softmax, validation['label'], validation], feed_dict={phase_train: False})

                        # Get metrics
                        Dixe = sdt.calc_metrics_segmentation(preds, labs, egs['patient'], dice_threshold=FLAGS.dice_threshold, batch_size=FLAGS.batch_size)

                        if Dixe:
                            DICE += Dixe
                            total += 1

                        # Convert inputs to numpy arrays
                        p11 = np.squeeze(preds.astype(np.float))
                        l11 = np.squeeze(labs.astype(np.float))
                        eg = np.squeeze(egs['image'].astype(np.float))
                        picd = []

                        for i in range(FLAGS.batch_size):

                            # Retreive one image, label and prediction from the batch to save
                            prediction = p11[i, :, :, 1]

                            # Manipulations to improve display data
                            lbl2 = np.copy(l11[i])  # Copy since we will print below
                            lbl2[lbl2 > 0] = 1  # For removing background noise in the image

                            # Zero out the background on the predicted map
                            prediction = np.multiply(np.squeeze(prediction), np.squeeze(lbl2))

                            # First create copies
                            p1 = np.copy(prediction)  # make an independent copy of the background nulled predictions
                            p2 = np.copy(l11[i])  # make an independent copy of labels map

                            # Now create boolean masks
                            p1[p1 > FLAGS.dice_threshold] = True  # Set predictions above threshold value to True
                            p1[p1 <= FLAGS.dice_threshold] = False  # Set those below to False
                            p2[p2 <= 1] = False  # Mark lung and background as False
                            p2[p2 > 1] = True  # Mark embolisms as True

                            # Check error
                            if np.sum(p1) > FLAGS.size_threshold and np.sum(p2) > 0: TP += 1
                            elif np.sum(p2) > 0 and np.sum(p1) < FLAGS.size_threshold: FN += 1
                            elif np.sum(p2) == 0 and np.sum(p1) < FLAGS.size_threshold: TN += 1
                            elif np.sum(p2) == 0 and np.sum(p1) > FLAGS.size_threshold: FP += 1
                            tot += 1

                        #     # Generate an overlay display
                        #     if np.sum(p2) > 5: picd.append(sdl.display_overlay(eg[i, 2], p1))
                        #
                        # # Show the images
                        # try: sdl.display_volume(np.asarray(picd), True)
                        # except: pass

                        # Garbage collection
                        del preds, labs, egs, eg, picd

                        # Increment step
                        step += 1

                        if step % 10 == 0: print ('Step %s of %s done' %(step, max_steps))

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Print final errors here
                    DICE_Score = DICE/total
                    print ('DICE Score: %s', (DICE_Score))
                    print('TP: %s, TN: %s, FP: %s, FN: %s, Slices: %s' % (TP, TN, FP, FN, tot))
                    print ('Sensitivity: %.2f %%, Specificity: %.2f %%' %((100*TP / (TP + FN)), (100* TN / (TN + FP))))

                    # Stop threads when done
                    coord.request_stop()

                    # Wait for threads to finish before closing session
                    coord.join(threads, stop_grace_period_secs=20)

                    # Shut down the session
                    mon_sess.close()

            # Break if this is the final checkpoint
            if 'Final' in Epoch: break

            # Print divider
            print('-' * 70)


def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(0)
    if tf.gfile.Exists(FLAGS.testing_dir):
        tf.gfile.DeleteRecursively(FLAGS.testing_dir)
    tf.gfile.MakeDirs(FLAGS.testing_dir)
    test()


if __name__ == '__main__':
    tf.app.run()