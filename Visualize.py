""" Training the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import os
import time
import numpy as np

import ILDModel as network
import tensorflow as tf
import SODTester as SDT
import SODLoader as SDL
import SOD_Display as SDD
import glob
from pathlib import Path

sdl = SDL.SODLoader(str(Path.home()) + '/PycharmProjects/Datasets/CT_Chest_ILD/')
sdd = SDD.SOD_Display()

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('data_dir', 'data/Viz/', """Path to the data directory.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_integer('box_dims', 40, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 40, """dimensions of the input pictures""")

# >5k example lesions total
tf.app.flags.DEFINE_integer('epoch_size', 18381, """Batch 3""")
tf.app.flags.DEFINE_integer('batch_size', 6127, """Number of images to process in a batch.""")

# Testing parameters
tf.app.flags.DEFINE_string('RunInfo', 'Viz/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 1, """Which GPU to use""")
tf.app.flags.DEFINE_float('cutoff', 14.0, """cutoff for percent of ILD wedges""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-3, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")


# Define a custom training class
def test():
    # Makes this the default graph where all ops will be added
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    with tf.Graph().as_default(), tf.device('/gpu:' + str(FLAGS.GPU)):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Get a dictionary of our images, id's, and labels from the iterator
        examples, iterator = network.inputs(False, skip=True)

        # Define input shape
        examples['data'] = tf.reshape(examples['data'], [FLAGS.batch_size, 10, FLAGS.network_dims, FLAGS.network_dims])

        # Display the images to tensorboard
        tf.summary.image('Test',
                         tf.reshape(examples['data'][0, 4, ...], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]),
                         8)

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

                # Print run info
                print("*** Vizualization Run %s on GPU %s ****" % (FLAGS.RunInfo, FLAGS.GPU))
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)
                mon_sess.run(var_init)
                mon_sess.run(iterator.initializer)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(mon_sess, ckpt.model_checkpoint_path)
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]
                else:
                    break

                # Initialize the step counter
                step = 0
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Tester instance
                sdt = SDT.SODTester(True, False)
                label_track, logit_track, pt_track = [], [], []

                try:
                    while step < max_steps:

                        # Load some metrics for testing
                        lbl1, logtz, pt, exs = mon_sess.run([labels, logits, examples['accno']],
                                                            feed_dict={phase_train: False})

                        # If first step then create the entry
                        if step == 0:
                            label_track = np.copy(lbl1)
                            logit_track = np.copy(logtz)
                            pt_track = np.copy(pt)
                        else:
                            label_track = np.concatenate([label_track, lbl1])
                            logit_track = np.concatenate([logit_track, logtz])
                            pt_track = np.concatenate([pt_track, pt])

                        # Increment step
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Calculate final MAE and ACC
                    data, lbl, logitz = combine_predictions_thresh(label_track, logit_track, pt_track, FLAGS.epoch_size,
                                                                   percent=FLAGS.cutoff)
                    sdt.calculate_metrics(logitz, lbl, 1, step)
                    sdt.retreive_metrics_classification(Epoch, True)
                    print('------ Current Best AUC: %.4f (Epoch: %s) --------' % (best_MAE, best_epoch))

                    # Shut down the session
                    mon_sess.close()

            # Break if this is the final checkpoint
            break


def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(0)
    if tf.gfile.Exists('testing/' + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo)
    tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo)
    test()


def combine_predictions_thresh(ground_truth, softmax, unique_ID, batch_size, percent=8.4, pos_cls=1, threshold=0.5):
    """
    Combines multi parametric predictions into one group
    :param ground_truth: raw labels from sess.run
    :param predictions: raw un-normalized logits
    :param unique_ID: a unique identifier for each patient (not example)
    :param batch_size: batch size
    :param percent: The percent above this indicate a positive
    :param pos_cls: the positive class
    :param threshold: Softmax threshold
    :return: recombined matrix, label array, logitz array
    """

    # Convert to numpy arrays
    predictions, label = np.squeeze(softmax.astype(np.float)), np.squeeze(ground_truth.astype(np.float))
    serz = np.squeeze(unique_ID)

    # The dictionary to return
    data = {}

    # Get the softmax scores
    sdt = SDT.SODTester(1, 0)
    predictions = sdt.calc_softmax_old(predictions)

    # add up the predictions
    for z in range(batch_size):

        # Calc
        if predictions[z, pos_cls] > threshold:
            pw = 1
        else:
            pw = 0

        # If we already have the entry then just append
        try:
            if serz[z] in data:
                data[serz[z]]['log0'] = data[serz[z]]['log0'] + predictions[z, 0]
                data[serz[z]]['log1'] = data[serz[z]]['log1'] + predictions[z, 1]
                data[serz[z]]['total'] += 1
                data[serz[z]]['pw'] += pw
            else:
                data[serz[z]] = {'label': label[z], 'log0': predictions[z, 0], 'log1': predictions[z, 1],
                                 'total': 1, 'avg': None, 'pw': pw}
        except:
            continue

    # Initialize new labels and logits
    logga, labba = [], []

    # Combine the data
    for idx, dic in data.items():

        # Get percentage positive
        dic['Pos_Percent'] = 100 * dic['pw'] / dic['total']

        # Calculate the new "logits
        if dic['Pos_Percent'] >= percent:
            avg = (0, 1)
        else:
            avg = (1, 0)

        # Append to trackers
        labba.append(dic['label'])
        logga.append(np.squeeze(avg))

        # add to the dictionary
        dic['avg'] = np.squeeze(avg)
        dic['ID'] = idx

        # TODO: Testing
        print('Acc: %s Lbl: %s, NPW: %.2f %% (%s), Tot: %s' % (
        idx, dic['label'], dic['Pos_Percent'], dic['pw'], dic['total']))

    return data, np.squeeze(labba), np.squeeze(logga)


if __name__ == '__main__':
    tf.app.run()