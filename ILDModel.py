# Defines and builds our network
#    Computes input images and labels using inputs() or distorted inputs ()
#    Computes inference on the models (forward pass) using inference()
#    Computes the total loss using loss()
#    Performs the backprop using train()

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

_author_ = 'simi'

import tensorflow as tf
import Input
import SODNetwork as SDN

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Retreive helper function object
sdn = SDN.SODMatrix()

def forward_pass(images, phase_train):

    """
    Train a 2 dimensional network
    :param images: input images, [batch, box_dims, box_dims, 3]
    :param phase_train: True if this is the training phase
    :return: L2 Loss and Logits
    """

    # Define each level
    base = tf.expand_dims(images[..., 0], -1)
    mid = tf.expand_dims(images[..., 1], -1)
    apex = tf.expand_dims(images[..., 2], -1)

    # Channel wise layers
    conva = sdn.convolution('Conva', apex, 3, 4, 2, phase_train=phase_train)
    convm = sdn.convolution('Convm', mid, 3, 4, 2, phase_train=phase_train)
    convb = sdn.convolution('Convb', base, 3, 4, 2, phase_train=phase_train)

    # Combine
    conv = conva + convm + convb
    conv = sdn.convolution('Pre_Conv', conv, 3, 8, 1, phase_train=phase_train)
    conv = sdn.convolution('Conv1', conv, 3, 16, 2, phase_train=phase_train)

    # Convolutional layers
    conv = sdn.residual_layer('Residual1', conv, 3, 32, phase_train=phase_train)
    conv = sdn.residual_layer('Residual1b', conv, 3, 32, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Residual2', conv, 3, 64, phase_train=phase_train)
    conv = sdn.inception_layer('Inception1', conv, 64, 1, phase_train=phase_train)
    conv = sdn.inception_layer('Inception2', conv, 128, 2, phase_train=phase_train)

    # Linear layers
    fc7 = sdn.fc7_layer('FC7a', conv, 16, True, phase_train, FLAGS.dropout_factor, override=3, BN=True)
    linear = sdn.linear_layer('Linear', fc7, 8, True, phase_train, FLAGS.dropout_factor, BN=True, relu=False)
    Logits = sdn.linear_layer('Softmax', linear, FLAGS.num_classes, relu=False, add_bias=False, BN=False)

    # Retreive the weights collection
    weights = tf.get_collection('weights')

    # Sum the losses
    L2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in weights]), FLAGS.l2_gamma)

    # Add it to the collection
    tf.add_to_collection('losses', L2_loss)

    # Activation summary
    tf.summary.scalar('L2_Loss', L2_loss)

    return Logits, L2_loss  # Return whatever the name of the final logits variable is


def total_loss(logits, labels):

    """
    Add loss to the trainable variables and a summary
        Args:
            logits: logits from the forward pass
            labels the true input labels, a 1-D tensor with 1 value for each image in the batch
        Returns:
            Your loss value as a Tensor (float)
    """

    # Apply cost sensitive loss here
    if FLAGS.loss_factor != 1.0:

        # Make a nodule sensitive binary for values > 1 in this case
        lesion_mask = tf.cast(labels >= FLAGS.loss_class, tf.float32)

        # Now multiply this mask by scaling factor then add back to labels. Add 1 to prevent 0 loss
        lesion_mask = tf.add(tf.multiply(lesion_mask, FLAGS.loss_factor), 1)

    # Change labels to one hot
    labels = tf.one_hot(tf.cast(labels, tf.uint8), depth=FLAGS.num_classes, dtype=tf.uint8)

    # Calculate  loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.squeeze(labels), logits=logits)

    # Apply cost sensitive loss here
    if FLAGS.loss_factor != 1.0: loss = tf.multiply(loss, tf.squeeze(lesion_mask))

    # Reduce to scalar
    loss = tf.reduce_mean(loss)

    # Output the summary of the MSE and MAE
    tf.summary.scalar('Cross Entropy', loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', loss)

    return loss


def backward_pass(total_loss):
    """ This function performs our backward pass and updates our gradients
    Args:
        total_loss is the summed loss caculated above
        global_step1 is the number of training steps we've done to this point, useful to implement learning rate decay
    Returns:
        train_op: operation for training"""

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Compute the gradients. NAdam optimizer came in tensorflow 1.2
    opt = tf.contrib.opt.NadamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1,
                                        beta2=FLAGS.beta2, epsilon=1e-8)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # clip the gradients
    #gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([train_op, variable_averages_op]):  # Wait until we apply the gradients
        dummy_op = tf.no_op(name='train')  # Does nothing. placeholder to control the execution of the graph

    return dummy_op


def inputs(filenames=[], training=True, skip=False):

    """
    Loads the inputs
    :param filenames: Filenames placeholder
    :param training: if training phase
    :param skip: Skip generating tfrecords if already done
    :return:
    """

    # To Do: Skip part 1 and 2 if the protobuff already exists
    if not skip: Input.pre_proc_25D(FLAGS.box_dims)

    else:
        print('-------------------------Previously saved records found! Loading...')

    return Input.load_protobuf(filenames, training)