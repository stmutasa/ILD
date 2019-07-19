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


def recurrent_convolution(self, scope, X, F, K, S=1, padding='SAME', phase_train=None, BN=True, bias=True, dropout=None):

    """
    This is a wrapper for convolutions
    :param scope:
    :param X: Output of the prior layer
    :param F: Convolutional filter size
    :param K: Number of feature maps
    :param S: Stride
    :param padding: 'SAME' or 'VALID'
    :param phase_train: For batch norm implementation
    :param BN: whether to perform batch normalization
    :param relu: bool, whether to do the activation function at the end
    :param downsample: whether to perform a max/avg downsampling at the end
    :param bias: whether to include a bias term
    :param dropout: whether to use dropout
    :return:
    """

    # Retreive number of inputs and channel size
    C = X.get_shape().as_list()[-1]
    Num = X.get_shape().as_list()[0]
    B = X.get_shape().as_list()[1]

    # Set the scope
    with tf.variable_scope(scope) as scope:

        # Set training phase variable
        self.training_phase = phase_train

        # Define the Kernel WHICH WE WILL USE FOR ALL CONVOLUTIONS
        kernel = tf.get_variable('Weights', shape=[F, F, C, K], initializer=tf.contrib.layers.variance_scaling_initializer())
        w_bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))

        # Add to the weights collection
        tf.add_to_collection('weights', kernel)
        tf.add_to_collection('biases', w_bias)

        # The initial convolution with these weights. Stride here
        conv = tf.nn.conv2d(X[0], kernel, [1, S, S, 1], padding=padding)
        if bias: conv = tf.nn.bias_add(conv, w_bias)
        if BN: conv = self.batch_normalization(conv, phase_train, scope)
        conv = tf.tanh(conv)

        # Add to state vector
        output_state = conv

        # Loop through the number of examples. Skipping the first which is done
        for z in range(1, Num):

            # Concat next input to state vector
            next_input = tf.concat([output_state, X[z]])

            # Calculate next output
            conv = tf.nn.conv2d(next_input, kernel, [1, 1, 1, 1], padding=padding)
            if bias: conv = tf.nn.bias_add(conv, w_bias)
            if BN: conv = self.batch_normalization(conv, phase_train, scope)
            conv = tf.tanh(conv)

            output_state = conv

        # Channel wise dropout
        if dropout and phase_train == True: conv = tf.nn.dropout(conv, dropout, noise_shape=[B, 1, 1, C])

        # Create a histogram/scalar summary of the conv1 layer
        if self.summary: self._activation_summary(conv)

        return conv

def forward_pass_dense(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    :param images: Images to analyze
    :param phase_train: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    # Define the Dense UNet
    dun = SDN.DenseUnet(nb_blocks=5, filters=6, images=images, phase_train=phase_train)

    # Now run the network
    conv = dun.define_network_25D(layers=[2, 4, 8, 16, 32], keep_prob=FLAGS.dropout_factor)

    # Output is a 1x1 box with 3 labels
    Logits = sdn.convolution('Logits', conv, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False)
    print('Logits: ', Logits)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def forward_pass_res(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    # First block: 3D conv -> Downsample (stried) -> Z Downsample
    print ('Input images: ', images)
    conv = sdn.residual_layer_3d('Conv1a', images, 3, 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer_3d('Conv1b', conv, 3, 8, 1, phase_train=phase_train)
    skip1 = sdn.convolution_3d('Skip1', conv, [5, 1, 1], 8, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv1c', conv, [2, 3, 3], 16, [1, 2, 2], 'VALID', phase_train) # 4x63x63x16
    print('*' * 30, conv)

    conv = sdn.residual_layer_3d('Conv2a', conv, 3, 16, 1, phase_train=phase_train)
    conv = sdn.residual_layer_3d('Conv2b', conv, 3, 16, 1, phase_train=phase_train)
    skip2 = sdn.convolution_3d('Skip2', conv, [4, 1, 1], 16, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv2c', conv, [2, 3, 3], 32, [1, 2, 2], 'VALID', phase_train)  # 3x31x31x32
    print('*' * 22, conv)

    conv = sdn.residual_layer_3d('Conv3a', conv, 3, 32, 1, phase_train=phase_train)
    conv = sdn.residual_layer_3d('Conv3b', conv, 3, 32, 1, phase_train=phase_train,)
    skip3 = sdn.convolution_3d('Skip3', conv, [3, 1, 1], 32, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv3c', conv, [2, 3, 3], 64, [1, 2, 2], 'VALID', phase_train)  # 2x15x15x15
    print('*'*14,conv)

    conv = sdn.inception_layer_3d('Conv4a', conv, 64, 1, phase_train=phase_train)
    conv = sdn.inception_layer_3d('Conv4b', conv, 64, 1, phase_train=phase_train)
    skip4 = sdn.convolution_3d('Skip4', conv, [2, 1, 1], 64, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv4c', conv, [2, 3, 3], 128, [1, 2, 2], 'VALID', phase_train)  # 1x7x7x128
    print('*'*6,conv)

    # From now on, we're 2D
    conv = tf.squeeze(conv)

    # Bottom of the decoder: 7x7
    conv = sdn.inception_layer('conv5_Inception', conv, 128, 1, 'SAME', phase_train, BN=False, relu=False)
    conv = sdn.residual_layer('Conv5', conv, 3, 128, 1, padding='SAME', phase_train=phase_train)
    conv = sdn.inception_layer('Conv5_Inception2', conv, 128, phase_train=phase_train)
    print('End Encoder: ', conv)

    # Upsample 1
    conv = sdn.deconvolution('Dconv1', conv, 3, 64, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip4), out_shape=[FLAGS.batch_size, 15, 15, 64])
    conv = sdn.inception_layer('Dconv1b', conv, 64, phase_train=phase_train)
    print('-'*6, conv)

    # Upsample 2
    conv = sdn.deconvolution('Dconv2', conv, 3, 32, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip3), out_shape=[FLAGS.batch_size, 31, 31, 32])
    conv = sdn.inception_layer('Dconv2b', conv, 32, phase_train=phase_train)
    print ('-'*14, conv)

    # Upsample 3
    conv = sdn.deconvolution('Dconv3', conv, 3, 16, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip2), out_shape=[FLAGS.batch_size, 63, 63, 16])
    conv = sdn.residual_layer('Dconv3b', conv, 3, 16, 1, phase_train=phase_train)
    print ('-'*22,conv)

    # Upsample 4
    conv = sdn.deconvolution('Dconv4', conv, 3, 8, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip1), out_shape=[FLAGS.batch_size, 128, 128, 8])
    conv = sdn.residual_layer('Dconv4b', conv, 3, 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Dconv4c', conv, 3, 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Dconv4d', conv, 3, 8, 1, phase_train=phase_train, dropout=FLAGS.dropout_factor)
    print ('-'*30, conv)

    # Output
    Logits = sdn.convolution('Logits', conv, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False)
    print('Logits: ', Logits)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def forward_pass_recurrent(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    # First block: 3D conv -> Downsample (stried) -> Z Downsample
    print ('Input images: ', images)
    conv = recurrent_convolution
    skip1 = sdn.convolution_3d('Skip1', conv, [5, 1, 1], 8, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv1c', conv, [2, 3, 3], 16, [1, 2, 2], 'VALID', phase_train) # 4x63x63x16
    print('*' * 30, conv)

    conv = sdn.residual_layer_3d('Conv2a', conv, 3, 16, 1, phase_train=phase_train)
    conv = sdn.residual_layer_3d('Conv2b', conv, 3, 16, 1, phase_train=phase_train)
    skip2 = sdn.convolution_3d('Skip2', conv, [4, 1, 1], 16, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv2c', conv, [2, 3, 3], 32, [1, 2, 2], 'VALID', phase_train)  # 3x31x31x32
    print('*' * 22, conv)

    conv = sdn.residual_layer_3d('Conv3a', conv, 3, 32, 1, phase_train=phase_train)
    conv = sdn.residual_layer_3d('Conv3b', conv, 3, 32, 1, phase_train=phase_train,)
    skip3 = sdn.convolution_3d('Skip3', conv, [3, 1, 1], 32, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv3c', conv, [2, 3, 3], 64, [1, 2, 2], 'VALID', phase_train)  # 2x15x15x15
    print('*'*14,conv)

    conv = sdn.inception_layer_3d('Conv4a', conv, 64, 1, phase_train=phase_train)
    conv = sdn.inception_layer_3d('Conv4b', conv, 64, 1, phase_train=phase_train)
    skip4 = sdn.convolution_3d('Skip4', conv, [2, 1, 1], 64, 1, 'VALID', phase_train, BN=False, relu=True)
    conv = sdn.convolution_3d('Conv4c', conv, [2, 3, 3], 128, [1, 2, 2], 'VALID', phase_train)  # 1x7x7x128
    print('*'*6,conv)

    # From now on, we're 2D
    conv = tf.squeeze(conv)

    # Bottom of the decoder: 7x7
    conv = sdn.inception_layer('conv5_Inception', conv, 128, 1, 'SAME', phase_train, BN=False, relu=False)
    conv = sdn.residual_layer('Conv5', conv, 3, 128, 1, padding='SAME', phase_train=phase_train)
    conv = sdn.inception_layer('Conv5_Inception2', conv, 128, phase_train=phase_train)
    print('End Encoder: ', conv)

    # Upsample 1
    conv = sdn.deconvolution('Dconv1', conv, 3, 64, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip4), out_shape=[FLAGS.batch_size, 15, 15, 64])
    conv = sdn.inception_layer('Dconv1b', conv, 64, phase_train=phase_train)
    print('-'*6, conv)

    # Upsample 2
    conv = sdn.deconvolution('Dconv2', conv, 3, 32, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip3), out_shape=[FLAGS.batch_size, 31, 31, 32])
    conv = sdn.inception_layer('Dconv2b', conv, 32, phase_train=phase_train)
    print ('-'*14, conv)

    # Upsample 3
    conv = sdn.deconvolution('Dconv3', conv, 3, 16, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip2), out_shape=[FLAGS.batch_size, 63, 63, 16])
    conv = sdn.residual_layer('Dconv3b', conv, 3, 16, 1, phase_train=phase_train)
    print ('-'*22,conv)

    # Upsample 4
    conv = sdn.deconvolution('Dconv4', conv, 3, 8, S=2, padding='VALID', phase_train=phase_train, concat=False,
                              concat_var=tf.squeeze(skip1), out_shape=[FLAGS.batch_size, 128, 128, 8])
    conv = sdn.residual_layer('Dconv4b', conv, 3, 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Dconv4c', conv, 3, 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Dconv4d', conv, 3, 8, 1, phase_train=phase_train, dropout=FLAGS.dropout_factor)
    print ('-'*30, conv)

    # Output
    Logits = sdn.convolution('Logits', conv, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False)
    print('Logits: ', Logits)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def forward_pass_peter_256(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    # First block: 3D conv -> Downsample (stried) -> Z Downsample
    print ('Input images: ', images) # 256
    conv1 = sdn.convolution_3d('Conv1a', images, [1, 3, 3], 8, 1, 'VALID', phase_train)  # 254
    skip1 = sdn.convolution_3d('Skip1', conv1, [5, 1, 1], 8, 1, 'VALID', phase_train, BN=False, relu=False)  # 254
    print ('*'*38, conv1)

    conv1 = sdn.convolution_3d('Conv1b', conv1, [2, 3, 3], 16, [1, 2, 2], 'VALID', phase_train) # 126
    conv1 = sdn.convolution_3d('Conv1c', conv1, [1, 3, 3], 16, 1, 'VALID', phase_train)  # 124
    skip2 = sdn.convolution_3d('Skip2', conv1, [4, 1, 1], 16, 1, 'VALID', phase_train, BN=False, relu=False)  # 124
    conv1 = sdn.convolution_3d('Conv1d', conv1, [1, 3, 3], 16, 1, 'VALID', phase_train)  # 122
    print ('*'*30, conv1)

    # 2nd block: 3D conv -> Downsample (stried) -> Z Downsample
    conv2 = sdn.convolution_3d('Conv2a', conv1, [2, 3, 3], 32, [1, 2, 2], 'VALID', phase_train) # 60
    skip3 = sdn.convolution_3d('Skip3', conv2, [3, 1, 1], 32, 1, 'VALID', phase_train, BN=False, relu=False)  # 60
    conv2 = sdn.convolution_3d('Conv2c', conv2, [1, 3, 3], 32, 1, 'VALID', phase_train) # 58
    print('*'*22,conv2)

    # 3rd block:
    conv3 = sdn.convolution_3d('Conv3a', conv2, [2, 3, 3], 64, [1, 2, 2], 'VALID', phase_train) # 28
    skip4 = sdn.convolution_3d('Skip4', conv3, [2, 1, 1], 64, 1, 'VALID', phase_train, BN=False, relu=False)  # 28
    conv3 = sdn.convolution_3d('Conv3c', conv3, [1, 3, 3], 64, 1, 'VALID', phase_train) # 26
    print('*'*14,conv3)

    # 4th block End
    conv4 = sdn.convolution_3d('Conv4a', conv3, [2, 3, 3], 128, [1, 2, 2], 'VALID', phase_train) # 12
    conv4 = sdn.convolution_3d('Conv4b', conv4, [1, 3, 3], 128, 1, 'VALID', phase_train) # 10
    conv4 = sdn.convolution_3d('Conv4c', conv4, [1, 3, 3], 128, 1, 'VALID', phase_train)  # 8
    print('End Encoder: ', conv4)

    conv4 = tf.squeeze(conv4)

    # Reverse 4th block
    conv5 = sdn.deconvolution('Conv5a', conv4, 3, 128, 1, 'VALID', phase_train, out_shape=[FLAGS.batch_size, 10, 10,128])
    conv5 = sdn.deconvolution('Conv5b', conv5, 3, 128, 1, 'VALID', phase_train, out_shape=[FLAGS.batch_size, 12, 12, 128])

    # Upsample 1
    conv6 = sdn.deconvolution('Dconv1a', conv5, 3, 64, 2, 'VALID', phase_train,out_shape=[FLAGS.batch_size, 26, 26, 64])
    conv6 = sdn.deconvolution('Dconv1b', conv6, 3, 64, 1, 'VALID', phase_train, False, tf.squeeze(skip4), out_shape=[FLAGS.batch_size, 28, 28, 64])
    print('-'*6, conv6)

    # Upsample 2
    conv7 = sdn.deconvolution('Dconv2a', conv6, 3, 32, 2, 'VALID', phase_train, out_shape=[FLAGS.batch_size, 58, 58, 32])
    conv7 = sdn.deconvolution('Dconv2b', conv7, 3, 32, 1, 'VALID', phase_train, False, tf.squeeze(skip3), out_shape=[FLAGS.batch_size, 60, 60, 32])
    print ('-'*14, conv7)

    # Upsample 3
    conv8 = sdn.deconvolution('Dconv3a', conv7, 3, 16, 2, 'VALID', phase_train, out_shape=[FLAGS.batch_size, 122, 122, 16])
    conv8 = sdn.deconvolution('Dconv3b', conv8, 3, 16, 1, 'VALID', phase_train, False, tf.squeeze(skip2), out_shape=[FLAGS.batch_size, 124, 124, 16])
    conv8 = sdn.deconvolution('Dconv3c', conv8, 3, 16, 1, 'VALID', phase_train, out_shape=[FLAGS.batch_size, 126, 126, 16])
    print ('-'*22,conv8)

    # Upsample 4
    conv9 = sdn.deconvolution('Dconv4a', conv8, 3, 8, 2, 'VALID', phase_train, False, tf.squeeze(skip1), out_shape=[FLAGS.batch_size, 254, 254, 8])

    # Output
    Logits = sdn.convolution('Logits', conv9, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False)
    print('Logits: ', Logits)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def total_loss(logitz, labelz, num_classes=2, class_weights=None, loss_type=None):

    """
    Cost function
    :param logitz: The raw log odds units output from the network
    :param labelz: The labels: not one hot encoded
    :param num_classes: number of classes predicted
    :param class_weights: class weight array
    :param loss_type: DICE or other to use dice or weighted
    :return:
    """

    # Reduce dimensionality
    labelz, logits = tf.squeeze(labelz), tf.squeeze(logitz)

    # Remove background label
    labels = tf.cast(labelz, tf.uint8)

    # Summary images
    imeg = int(FLAGS.batch_size/2)
    tf.summary.image('Labels', tf.reshape(tf.cast(labels[imeg], tf.float32), shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)
    tf.summary.image('Logits', tf.reshape(logitz[imeg,:,:,1], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)

    if loss_type=='DICE_SPARSE':

        # Flatten
        prediction = tf.reshape(logitz, [-1, num_classes])
        labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)

        # Cast labels to int64 [0, 1, 1, ..0]
        labels = tf.to_int64(labels)

        # Calculate softmax of the prediction vector at the -1 axis
        prediction = tf.nn.softmax(prediction)

        # Create a sequence of numbers from 0 to every pixel [0, 1, .., n]
        ids = tf.range(tf.to_int64(tf.shape(labels)[0]), dtype=tf.int64)

        # Stacks rank n tensors into rank n+1 tensors: [[id1, lab1], [id2, lab2], ..[idn,labn]]
        ids = tf.stack([ids, labels], axis=1)

        # Create a sparse tensor. Indices are the entries with nonzero entries. Values is the value of each element
        one_hot = tf.SparseTensor(indices=ids, values=tf.ones_like(labels, dtype=tf.float32),
                                  dense_shape=tf.to_int64(tf.shape(prediction)))

        # Now calculate the numerator
        dice_numerator = 2.0 * tf.sparse_reduce_sum(one_hot * prediction, reduction_axes=[0])

        # Calculate the denominator
        dice_denominator = tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
                           tf.sparse_reduce_sum(one_hot, reduction_axes=[0])

        # To prevent math errors
        epsilon_denominator = 0.00001

        # Calculate the DICE score
        dice_score = dice_numerator / (dice_denominator + epsilon_denominator)

        # Now get the dice score
        loss =  1.0 - tf.reduce_mean(dice_score)

    elif loss_type=='DICE':

        # Make labels one hot
        labels = tf.cast(tf.one_hot(labels, depth=FLAGS.num_classes, dtype=tf.uint8), tf.float32)

        # # Generate and apply mask
        # mask = tf.expand_dims(tf.cast(labelz > 0, tf.float32), -1)
        # logits, labels = logitz * mask, labels * mask

        # Flatten
        logits = tf.reshape(logits, [-1, num_classes])
        labels = tf.reshape(labels, [-1, num_classes])

        # To prevent number errors:
        eps = 1e-5

        # Calculate softmax:
        logits = tf.nn.softmax(logits)

        # Find the intersection
        intersection = 2*tf.reduce_sum(logits * labels)

        # find the union
        union = eps + tf.reduce_sum(logits) + tf.reduce_sum(labels)

        # Calculate the loss
        dice = intersection/union

        # Output the training DICE score
        tf.summary.scalar('DICE_Score', dice)

        # 1-DICE since we want better scores to have lower loss
        loss = 1 - dice

    else:

        # Make labels one hot
        labels = tf.cast(tf.one_hot(labels, depth=FLAGS.num_classes, dtype=tf.uint8), tf.float32)

        # # Generate and apply mask
        # mask = tf.expand_dims(tf.cast(labelz > 0, tf.float32), -1)
        # logits, labels = logitz * mask, labels * mask

        # Generate class weights
        class_weights = tf.Variable([1, FLAGS.loss_factor], trainable=False)

        # Flatten
        logits = tf.reshape(logits, [-1, num_classes])
        labels = tf.cast(tf.reshape(labels, [-1, num_classes]), tf.float32)

        # Make our weight map
        weight_map = tf.multiply(labels, class_weights)
        weight_map = tf.reduce_sum(weight_map, axis=1)

        # Calculate the loss: Result is batch x 65k
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        # Apply the class weights
        loss = tf.multiply(loss, weight_map)

        # Reduce the loss into a scalar
        loss = tf.reduce_mean(loss)

    # Output the Loss
    tf.summary.scalar('Loss_Raw', loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', loss)

    return loss


def backward_pass(total_loss):

    """
    Perform the backward pass and update the gradients
    :param total_loss:
    :return:
    """

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.train.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Compute the gradients. NAdam optimizer came in tensorflow 1.2
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables(): tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # Control graph execution
    with tf.control_dependencies([train_op, variable_averages_op]):  dummy_op = tf.no_op(name='train')

    return dummy_op


def inputs(skip=False):

    """
    Load the raw inputs
    :param skip:
    :return:
    """

    # Skip part 1 and 2 if the protobuff already exists
    if not skip:
        Input.pre_proc_25D(FLAGS.slice_gap, FLAGS.box_dims)

    else: print('-------------------------Previously saved records found! Loading...')

    # Part 2: Load the protobuff  -----------------------------
    print('----------------------------------------Loading Protobuff...')
    train = Input.load_protobuf()
    valid = Input.load_validation()


    return train, valid