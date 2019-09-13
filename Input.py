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
def pre_proc_wedge_3d(dims=512, size=[10, 40, 40], stride=[7, 25, 25]):

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
        midlung_slice = inf + ((sup - inf) // 2)

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

        # Done with patient, if there are no wedges saved, reduce the stride
        counts = index - counts
        print('***** Pt %s with %s counts' % (MRN, counts))

        if counts <= 50:
            print('\nCounts too low  for %s, (%s) with dims %s spacing %s stride div %s and trying again..'
                  % (MRN, counts, volume.shape, spacing, ((100 - counts) // 30)), end='')
            index -= counts  # Overrwrite the already written data
            thresh = 0.45
            if counts == 0:
                true_stride = (true_stride // 5).astype(np.int16)
                thresh = 0.25
            else:
                true_stride = (true_stride // ((100 - counts) // 30)).astype(np.int16)
            counts = index
            for z in range(inf, midlung_slice, true_stride[0]):
                for y in range(0, volume.shape[1], true_stride[1]):
                    for x in range(0, volume.shape[2], true_stride[2]):
                        wedge_check, _ = sdl.generate_box(mask, [z, y, x], box_size[1], z_overwrite=box_size[0])
                        ratio = sdl.return_nonzero_pixel_ratio(wedge_check, 1, True)
                        if (ratio < thresh) or (ratio > 0.99999): continue
                        wedge, _ = sdl.generate_box(volume, [z, y, x], box_size[1], z_overwrite=box_size[0])
                        wedge, _ = sdl.resample(wedge, spacing, new_spacing=[1, 1, 1])
                        wedge = sdl.resize_volume(wedge, np.int16, size[2], size[1], size[0])
                        data[index] = {'data': wedge, 'label': label, 'label_raw': label_raw, 'accno': Accno,
                                       'MRN': MRN, 'file': file, 'sizexy': size[1], 'sizez': size[0]}
                        index += 1
                        del wedge, wedge_check
            counts = index - counts
            print(' Made %s this time\n' % counts)
            if counts == 0: sdd.display_whole_vol_overlay(volume, mask)

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
    plt.show()


# Load the protobuf
def load_protobuf(filenames, training=True):

    """
    Loads the protocol buffer into a form to send to shuffle
    """

    # Create a dataset from the protobuf
    dataset = tf.data.TFRecordDataset(filenames)

    _records_call = lambda dataset: \
        sdl.load_tfrecords(dataset, [10, FLAGS.box_dims, FLAGS.box_dims], tf.int16)

    # Parse the record into tensors
    dataset = dataset.map(_records_call, num_parallel_calls=6)

    # Warp the data set
    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=6)

    # Repeat input indefinitely
    dataset = dataset.repeat()

    # Shuffle the dataset then create a batch
    if training: dataset = dataset.shuffle(buffer_size=1000)
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
    image = record['data']

    if self._distords:  # Training

        # Data Augmentation ------------------ Contrast, brightness, noise, rotate, shear, crop, flip

        # Now normalize. Window level is -600, width is 1500
        image += 600
        image /= 1500

        # Then randomly flip
        image = tf.image.random_flip_left_right(tf.image.random_flip_up_down(image))

        # Random brightness/contrast
        image = tf.image.random_contrast(image, lower=0.995, upper=1.005)

        # Reshape image to be a factorizatoin of 2
        image = sdl.tf_resize_3D(image, 8, FLAGS.network_dims, FLAGS.network_dims, True)

        # For noise, first randomly determine how 'noisy' this study will be
        T_noise = tf.random_uniform([1], 0, 0.1)

        # Create a poisson noise array
        noise = tf.random_uniform(shape=[8, FLAGS.network_dims, FLAGS.network_dims], minval=-T_noise, maxval=T_noise)

        # Add the poisson noise
        image = tf.add(image, tf.cast(noise, tf.float32))

    else: # Validation

        # Now normalize. Window level is -600, width is 1500
        image += 600
        image /= 1500

        # Reshape image to factor of 2
        image = tf.image.resize_images(image, [8, FLAGS.network_dims, FLAGS.network_dims])

    # Make record image
    record['data'] = image

    return record

# pre_proc_wedge_3d()
