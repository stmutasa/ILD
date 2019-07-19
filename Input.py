"""
Load and preprocess the files to a protobuff
"""

import os

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD

from pathlib import Path
from distutils.util import strtobool
from random import shuffle
import matplotlib

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/CT_Chest_ILD/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()

# For loading the files for a 2.5 D network
def pre_proc_25D(slice_gap=2, dims=256):

    """
    Loads the CT data into a tfrecords file
    :param slice_gap: the gap (mm) between slices to save for 3D data
    :param dims: the dimensions of the images saved
    :return:
    """

    # First retreive the filenames
    filenames = sdl.retreive_filelist('*', path=home_dir, include_subfolders=True)
    shuffle(filenames)

    # global variables
    index, pts, per = 0, 0, 0
    data = {}

    # Loop through all the files
    for file in filenames:

        # Now load the volumes
        try: volume, orig, spacing, _, header = sdl.load_DICOM_3D(file, return_header=True)
        except:
            print ('Unable to load: ', file)
            continue

        # Generate a lung mask
        mask = sdl.create_lung_mask(volume)

        # TODO: Testing
        sdd.display_volume(volume)
        sdd.display_volume(mask)
        sdd.display_vol_label_overlay(volume, mask, title='Test', display_non=True, plot=True)

        # Apply a lung mask overlay
        overlay = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], 3], np.float32)
        for z in range(volume.shape[0]): overlay[z] = sdl.display_overlay(volume[z], mask[z])

        # TODO: Testing
        sdl.display_volume(volume)
        sdl.display_volume(overlay)
        print (volume.shape, spacing)

    #     # Window the volume and center it at 0
    #     volume = sdl.window_image(volume, 40, 20)
    #     volume -= 40
    #
    #     # Generate labels, hemorrhage = 1, edema = 2, other = 3
    #     segments = bleed_segment + edema_segment
    #     segments [segments>2] = 1
    #
    #     # Loop through the image volume
    #     for z in range (volume.shape[0]):
    #
    #         # Calculate a scaled slice shift
    #         sz = 1
    #
    #         # Skip very bottom and very top of image
    #         if ((z-3*sz) < 0) or ((z+3*sz) > volume.shape[0]): continue
    #
    #         # Label is easy, just save the slice
    #         data_label = sdl.zoom_2D(segments[z].astype(np.int16), (dims, dims))
    #
    #         # Generate the empty data array
    #         data_image = np.zeros(shape=[5, dims, dims], dtype=np.int16)
    #
    #         # Set starting point
    #         zs = z - (2*sz)
    #
    #         # Save 5 slices with shift Sz
    #         for s in range(5): data_image[s, :, :] = sdl.zoom_2D(volume[zs+(s*sz)].astype(np.int16), [dims, dims])
    #
    #         # If there is label here, save more the slices
    #         sum_check = np.sum(np.squeeze(data_label > 0).astype(np.uint8))
    #         if sum_check > 5: num_egs = 2
    #         else: num_egs = 1
    #
    #         for _ in range(num_egs):
    #
    #             # Save the dictionary: int16, int16, int, int
    #             data[index] = {'image_data': data_image, 'label_data': data_label, 'patient': patient, 'slice': z, 'study': study}
    #
    #             # Finished with this slice
    #             index += 1
    #
    #         # Garbage collection
    #         del data_label, data_image
    #
    #     # Finished with all of this patients embolisms
    #     pts += 1
    #
    #     # Save every 7 patients
    #     if pts %30 == 0:
    #
    #         # Counters
    #         per = index-per
    #
    #         print ('%s Patients loaded, %s slices saved (%s this protobuf)' %(pts, index, per))
    #
    #         sdl.save_tfrecords(data, 1, 0, file_root=('data/Edema%s' %int(pts/30)))
    #         if pts < 35: sdl.save_dict_filetypes(data[0])
    #
    #         del data
    #         data = {}
    #
    # # Finished with all the patients
    # if len(data)>0: sdl.save_tfrecords(data, 1, 'data/EdemaFin')
    # print ('%s Total patients loaded, %s total slices. Final protobuf: %s slices' %(pts, index, len(data)))
    sdd.display_volume(volume, True)


# Load the protobuf
def load_protobuf():

    """
    Loads the protocol buffer into a form to send to shuffle
    """

    # Load all the filenames in glob
    filenames1 = sdl.retreive_filelist('tfrecords', False, path='data/')
    filenames = []

    # Define the filenames to remove
    for i in range (0, len(filenames1)):
        if FLAGS.test_files not in filenames1[i]:
            filenames.append(filenames1[i])

    # Show the file names
    print('Training files: %s' % filenames)

    # now load the remaining files
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims, tf.int16, z_dim=5, segments='label_data')

    print (data['image_data'], data['label_data'])

    # Image augmentation. First calc rotation parameters
    angle = tf.random_uniform([1], -0.52, 0.52)
    data['image_data'] = tf.add(data['image_data'], 200.0)

    # Random rotate
    data['image_data'] = tf.contrib.image.rotate(data['image_data'], angle)
    data['label_data'] = tf.contrib.image.rotate(data['label_data'], angle)

    # Return image to center
    data['image_data'] = tf.subtract(data['image_data'], 200.0)

    # Random gaussian noise
    data['image_data'] = tf.image.random_brightness(data['image_data'], max_delta=5)
    data['image_data'] = tf.image.random_contrast(data['image_data'], lower=0.95, upper=1.05)

    # Reshape image
    data['image_data'] = tf.image.resize_images(data['image_data'], [FLAGS.network_dims, FLAGS.network_dims])
    data['label_data'] = tf.image.resize_images(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims])

    # For noise, first randomly determine how 'noisy' this study will be
    T_noise = tf.random_uniform([1], 0, FLAGS.noise_threshold)

    # Create a poisson noise array
    noise = tf.random_uniform(shape=[5, FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)

    # Add the gaussian noise
    data['image_data'] = tf.add(data['image_data'], tf.cast(noise, tf.float32))

    # Display the images
    tf.summary.image('Train IMG', tf.reshape(data['image_data'][2], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)
    tf.summary.image('Train Label IMG', tf.reshape(data['label_data'], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)

    # Return data as a dictionary
    return sdl.randomize_batches(data, FLAGS.batch_size)


# Load the validation set
def load_validation():
    """
    Loads the protocol buffer into a form to send to shuffle
    :param
    :return:
    """

    # Load all the filenames in glob
    filenames1 = sdl.retreive_filelist('tfrecords', False, path='data/')
    filenames = []

    # Define the filenames to remove
    for i in range(0, len(filenames1)):
        if FLAGS.test_files in filenames1[i]:
            filenames.append(filenames1[i])

    # Show the file names
    print('Testing files: %s' % filenames)

    # now load the remaining files
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims, tf.int16, z_dim=5, segments='label_data')

    # Reshape image
    data['image_data'] = tf.image.resize_images(data['image_data'], [FLAGS.network_dims, FLAGS.network_dims])
    data['label_data'] = tf.image.resize_images(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims])

    # Display the images
    tf.summary.image('Test IMG', tf.reshape(data['image_data'][2], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)
    tf.summary.image('Test Label IMG', tf.reshape(data['label_data'], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)

    # Return data as a dictionary
    return sdl.val_batches(data, FLAGS.batch_size)


pre_proc_25D()