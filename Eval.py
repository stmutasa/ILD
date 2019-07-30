""" Training the network on a single GPU """

from __future__ import absolute_import      # import multi line and Absolute/Relative
from __future__ import division             # change the division operator to output float if dividing two integers
from __future__ import print_function       # use the print function from python 3

import os
import time                                 # to retreive current time
import numpy as np

import ILDModel as network
import tensorflow as tf
import SODTester as SDT
import SODLoader as SDL
import SOD_Display as SDD
import glob

sdl = SDL.SODLoader(data_root='/PycharmProjects/Datasets/BreastData/MRI/iSpy/')
sdd = SDD.SOD_Display()

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('data_dir', 'data/', """Path to the data directory.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_string('test_files', '_9', """Files for testing have this name""")
tf.app.flags.DEFINE_integer('box_dims', 512, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 128, """dimensions of the input pictures""")

# >5k example lesions total
tf.app.flags.DEFINE_integer('epoch_size', 400, """How many images were loaded""")
tf.app.flags.DEFINE_integer('batch_size', 400, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-3, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")
tf.app.flags.DEFINE_float('loss_factor', 1.0, """Penalty for missing a class is this times more severe""")
tf.app.flags.DEFINE_integer('loss_class', 1, """For classes this and above, apply the above loss factor.""")

# Hyperparameters to control the optimizer
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")

# Multi GPU Training parameters
tf.app.flags.DEFINE_string('RunInfo', 'Run2/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")

# Define a custom training class
def test():


    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Use a placeholder for the filenames
        filenames = tf.placeholder(tf.string, shape=[None])

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Get a dictionary of our images, id's, and labels from the iterator
        examples, iterator = network.inputs(filenames, True, skip=True)

        # Define input shape
        examples['data'] = tf.reshape(examples['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims, 3])

        # Display the images
        tf.summary.image('Base Val', tf.reshape(examples['data'][0, :, :, 0], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)
        tf.summary.image('Mid Val', tf.reshape(examples['data'][0, :, :, 1], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)
        tf.summary.image('Apex Val', tf.reshape(examples['data'][0, :, :, 2], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, l2loss = network.forward_pass(examples['data'], phase_train=phase_train)

        # Labels
        labels = examples['label']

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # Trackers for best performers
        best_MAE, best_epoch = 0.25, 0

        # Tester instance
        sdt = SDT.SODTester(True, False)

        while True:

            # Allow memory placement growth
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as mon_sess:

                # Define filenames
                all_files = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
                valid_files = [x for x in all_files if FLAGS.test_files in x]

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir+FLAGS.RunInfo)

                # Initialize the variables
                mon_sess.run(var_init)

                # Initialize iterator
                mon_sess.run(iterator.initializer, feed_dict={filenames: valid_files})

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the model
                    saver.restore(mon_sess, ckpt.model_checkpoint_path)

                    # Extract the epoch
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('Epoch')[-1]

                else:
                    print ('No checkpoint file found')
                    break

                # Initialize the step counter
                step = 0

                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Tester instance
                sdt = SDT.SODTester(True, False)

                try:
                    while step < max_steps:

                        # Load some metrics for testing
                        lbl1, logtz, pt = mon_sess.run([labels, logits, examples['accno']], feed_dict={phase_train: False})

                        # Increment step
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Calculate final MAE and ACC
                    # data, labz, logz = sdt.combine_predictions(lbl1, logtz, pt, FLAGS.batch_size)
                    # sdt.calculate_metrics(logz, labz, 1, step)
                    sdt.calculate_metrics(logtz, lbl1, 1, step)
                    sdt.retreive_metrics_classification(Epoch, True)
                    print('------ Current Best AUC: %.4f (Epoch: %s) --------' % (best_MAE, best_epoch))

                    # Lets save runs that perform well
                    if sdt.AUC >= best_MAE:

                        # Save the checkpoint
                        print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                        # Define the filenames
                        checkpoint_file = os.path.join('testing/' + FLAGS.RunInfo, ('Epoch_%s_AUC_%0.3f' % (Epoch, sdt.AUC)))
                        csv_file = os.path.join('testing/' + FLAGS.RunInfo, ('%s_E_%s_AUC_%0.2f.csv' % (FLAGS.RunInfo[:-1], Epoch, sdt.AUC)))

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)
                        #sdl.save_Dict_CSV(data, csv_file)

                        # Save a new best MAE
                        best_MAE = sdt.AUC
                        best_epoch = Epoch

                    # Shut down the session
                    mon_sess.close()

            # Break if this is the final checkpoint
            if '299' in Epoch: break

            # Print divider
            print('-' * 70)

            # Otherwise check folder for changes
            filecheck = glob.glob(FLAGS.train_dir+FLAGS.RunInfo + '*')
            newfilec = filecheck

            # Sleep if no changes
            while filecheck == newfilec:

                # Sleep an amount of time proportional to the epoch size
                time.sleep(int(FLAGS.epoch_size * 0.05))

                # Recheck the folder for changes
                newfilec = glob.glob(FLAGS.train_dir+FLAGS.RunInfo + '*')



def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(600)
    if tf.gfile.Exists('testing/'):
        tf.gfile.DeleteRecursively('testing/')
    tf.gfile.MakeDirs('testing/')
    test()


if __name__ == '__main__':
    tf.app.run()
