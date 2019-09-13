"""
Load and preprocess the files to a protobuff
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD

from pathlib import Path
from random import shuffle
import matplotlib.pyplot as plt

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/CT_Chest_ILD/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()

# For loading the files for a 2.5 D network
def pre_proc_wedge_3d(dims=512, size=[10, 40, 40], stride=[5, 20, 20]):

    """
    Loads the CT data into tfrecords
    This version creates virtual wedge biopsies
    :param dims:
    :param size:
    :param stride:
    :return:
    """

    # First retreive the filenames
    filenames = sdl.retreive_filelist('*', path=home_dir, include_subfolders=True)
    shuffle(filenames)

    # retrieve the labels
    label_file = sdl.retreive_filelist('csv', path=home_dir, include_subfolders=True)[0]
    labels = sdl.load_CSV_Dict('Accno', label_file)

    # global variables
    index, pts, per = 0, 0, 0
    data, track = {}, {}
    display, failures = [], [0, 0, 0]

    # Loop through all the files
    for file in filenames:

        # Now load the volumes
        try:
            volume, _, spacing, _, header = sdl.load_DICOM_3D(file, return_header=True, sort='Lung', display=True)
        except:
            print ('Unable to load: ', file, '\n')
            failures[2] +=1
            continue

        # Retreive pt info
        Accno = header['tags'].AccessionNumber
        MRN = header['tags'].PatientID
        try: label_raw = int(labels[Accno]['Label'])
        except:
            failures[0] +=1
            del volume
            continue

        # Fix labels per Hiram
        if label_raw < 2: label = 1
        elif label_raw == 3: label = 0
        else:
            failures[1] += 1
            del volume
            continue

        # Display volumes accidentally loaded as coronal
        if volume.shape[1] != volume.shape[2]:
            print('\n ********* Pt %s weird shaped %s\n' % (Accno, volume.shape))

        # Fix strange z axis spacing
        if spacing[0] > 10: spacing[0] = 1

        # Resize the volume
        if volume.shape[1] != dims:
            volume = sdl.resize_volume(volume, np.int16, dims, dims)

        # Generate a lung mask
        mask = sdl.create_lung_mask(volume, close=12, dilate=15)

        # window the lungs
        volume = sdl.window_image(volume, -600, 750)

        # Calculate the inferior extent of the mask
        for slice in range(mask.shape[0]):

            # Check if this slice of the mask has label
            slice_img = mask[slice]
            if np.sum(slice_img) > 0:
                inf = slice
                break

        # Calculate superior extent
        for slice in range(mask.shape[0] - 1, 0, -1):

            # Check if this slice of the mask has label
            slice_img = mask[slice]
            if np.sum(slice_img) > 0:
                sup = slice
                break

        # If the superior extent is max, we likely have trachea, add 10 mm
        if sup > volume.shape[0] - 3:
            sup = int(sup - 10 // spacing[0])

        # Find the middle of the lungs
        midlung_slice = (sup - inf) // 2

        # Calculate how big the box should be
        box_size = (np.asarray(size) // spacing).astype(np.int16)
        true_stride = (np.asarray(stride) // spacing).astype(np.int16)

        # For counting
        counts = index

        # Loop through and create the wedges
        for z in range(inf, midlung_slice, true_stride[0]):
            for y in range(0, volume.shape[1], true_stride[1]):
                for x in range(0, volume.shape[2], true_stride[2]):

                    # Create a segment wedge here
                    wedge_check, _ = sdl.generate_box(mask, [z, y, x], box_size[1], z_overwrite=box_size[0])

                    # Check if there is > 50% lung here, if not, discard and continue
                    ratio = sdl.return_nonzero_pixel_ratio(wedge_check, 1, True)
                    if (ratio < 0.5) or (ratio > 0.99999): continue

                    # Sucess, make a wedge!
                    wedge, _ = sdl.generate_box(volume, [z, y, x], box_size[1], z_overwrite=box_size[0])

                    # Resample the wedge
                    wedge, _ = sdl.resample(wedge, spacing, new_spacing=[1, 1, 1])
                    wedge = sdl.resize_volume(wedge, np.int16, size[2], size[1], size[0])

                    # Save
                    data[index] = {'data': wedge, 'label': label, 'label_raw': label_raw, 'accno': Accno, 'MRN': MRN,
                                   'file': file, 'sizexy': size[1], 'sizez': size[0]}
                    index += 1

                    # Garbage
                    del wedge, wedge_check

        # Done with patient
        counts = index - counts
        display.append(counts)
        pts += 1
        del volume, mask

        # Save every 20 patients
        if pts % 40 == 0:
            print('%s Patients complete, %s Wedges saved' % (pts, index))
            print('Wedges per in this protobuf: \n%s' % display)
            file_root = ('data/Egs_' + str(pts // 40))
            sdl.save_tfrecords(data, 1, file_root=file_root)
            if pts < 45: sdl.save_dict_filetypes(data[0])
            del data, display
            data, display = {}, []

    # All patients done, print the summary message
    print('%s Patients saved, %s failed[No label, Label out of range, Failed load] %s' % (pts, sum(failures), failures))

    # Now create a final protocol buffer
    print('Creating final protocol buffer')
    if data:
        print('%s patients complete, %s images saved' % (pts, index))
        print('Patients in this protobuf: \n%s' % display)
        sdl.save_tfrecords(data, 1, file_root='data/Egs_Fin')
        del data, display


# Load the protobuf
def load_protobuf(filenames, training=True):

    """
    Loads the protocol buffer into a form to send to shuffle
    """

    # Create a dataset from the protobuf
    dataset = tf.data.TFRecordDataset(filenames)

    if training:
        _records_call = lambda dataset: \
            sdl.load_tfrecords(dataset, [10, FLAGS.box_dims, FLAGS.box_dims, 3], tf.int16)

    else:
        _records_call = lambda dataset: \
            sdl.load_tfrecords(dataset, [FLAGS.box_dims, FLAGS.box_dims, 3], tf.int16)

    # Parse the record into tensors
    dataset = dataset.map(_records_call, num_parallel_calls=6)

    # Warp the data set
    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=6)

    # Repeat input indefinitely
    dataset = dataset.repeat()

    # Shuffle the dataset then create a batch
    if training: dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(FLAGS.batch_size)

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Retreive the batch
    examples = iterator.get_next()

    # Return data as a dictionary
    return examples, iterator


class DataPreprocessor(object):

    # Applies transformations to dataset

  def __init__(self, distords):
    self._distords = distords

  def __call__(self, record):

    """Process img for training or eval."""
    apex, midlung, base = record['data'][...,0], record['data'][...,1], record['data'][...,2]

    if self._distords:  # Training

        # Data Augmentation ------------------ Contrast, brightness, noise, rotate, shear, crop, flip

        # Generate random slices to use
        slice_a = tf.squeeze(tf.random_uniform([1], 0, 10, dtype=tf.int32))
        slice_m = tf.squeeze(tf.random_uniform([1], 0, 10, dtype=tf.int32))
        slice_b = tf.squeeze(tf.random_uniform([1], 0, 10, dtype=tf.int32))

        # Apply the slices
        apex, midlung, base = tf.squeeze(apex[slice_a]), tf.squeeze(midlung[slice_m]), tf.squeeze(base[slice_b])

        # Stack the results on a per channel basis
        image = tf.stack([apex, midlung, base], -1)

        # Now normalize. Window level is -600, width is 1500
        image += 600
        image /= 1500

        # Image augmentation. First calc rotation parameters
        angle = tf.random_uniform([1], -0.30, 0.30)

        # Random rotate
        image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')

        # Then randomly flip
        image = tf.image.random_flip_left_right(tf.image.random_flip_up_down(image))

        # Random brightness/contrast
        image = tf.image.random_contrast(image, lower=0.95, upper=1.05)

        # Random center crop
        image = tf.image.central_crop(image, 0.8)

        # Reshape image
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims])

        # For noise, first randomly determine how 'noisy' this study will be
        T_noise = tf.random_uniform([1], 0, 0.1)

        # Create a poisson noise array
        noise = tf.random_uniform(shape=[FLAGS.network_dims, FLAGS.network_dims, 3], minval=-T_noise, maxval=T_noise)

        # Add the poisson noise
        image = tf.add(image, tf.cast(noise, tf.float32))

    else: # Validation

        # Apply the slices
        apex, midlung, base = tf.squeeze(apex), tf.squeeze(midlung), tf.squeeze(base)

        # Stack the results on a per channel basis
        image = tf.stack([apex, midlung, base], -1)

        # Now normalize. Window level is -600, width is 1500
        image += 600
        image /= 1500

       # Center crop
        image = tf.image.central_crop(image, 0.8)

        # Reshape image
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims])

    # Make record image
    record['data'] = image

    return record

# pre_proc_25D()
# filenames = tf.placeholder(tf.string, shape=[None])
# load_protobuf(filenames)
pre_proc_wedge_3d()
