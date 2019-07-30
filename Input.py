"""
Load and preprocess the files to a protobuff
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD

from pathlib import Path
from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/CT_Chest_ILD/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()

# For loading the files for a 2.5 D network
def pre_proc_25D(dims=512):

    """
    Loads the CT data into a tfrecords file
    :param slice_gap: the gap (mm) between slices to save for 3D data
    :param dims: the dimensions of the images saved
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
    data = {}
    display, failures = [], [0, 0, 0]

    # Loop through all the files
    for file in filenames:

        # Now load the volumes
        try: volume, orig, spacing, _, header = sdl.load_DICOM_3D(file, return_header=True, sort='Lung', display=True)
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
        if volume.shape[1] != volume.shape[2]: print('\n ********* Pt %s weird shaped %s\n' % (Accno, volume.shape))

        # TODO: Testing
        print(volume.shape, Accno, MRN, label_raw, label)
        # volume = sdl.window_image(volume, -600, 750)
        # sdd.display_volume(volume, True)

        # Resize the volume
        if volume.shape[1] != dims:
            volume = sdl.resize_volume(volume, np.int16, dims, dims)

        # Generate a lung mask
        mask = sdl.create_lung_mask(volume)

        # window the lungs
        volume = sdl.window_image(volume, -600, 750)

        # Calculate the superior extent of the mask
        for slice in range(mask.shape[0]):

            # Check if this slice of the mask has label
            slice_img = mask[slice]
            if np.sum(slice_img) > 0:
                sup = slice
                break

        # Calculate inferior extent
        for slice in range(mask.shape[0] - 1, 0, -1):

            # Check if this slice of the mask has label
            slice_img = mask[slice]
            if np.sum(slice_img) > 0:
                inf = slice
                break

        # Find the middle of the thirds
        total = inf - sup
        apex = sup + (total / 3) // 2
        mid = sup + (2 * total // 3 - total / 3 // 2)
        lower = sup + (total - total / 3 // 2)
        spacing = total/3/2//10

        # Create the empty arrays we will fill
        apical = np.zeros(shape=[10, dims, dims], dtype=np.int16)
        midlung, lungbase = np.zeros_like(apical), np.zeros_like(apical)

        # Now save the segments. Loop every 3 slices
        for z in range (10):

            # Apical segment, start at -spacing
            apical_start = int((apex-spacing) + (z*spacing))
            apical[z] = volume[apical_start]

            # Midlung segment, start at -spacing*5
            midlung_start = int((mid - spacing*5) + (z * spacing))
            midlung[z] = volume[midlung_start]

            # Apical segment, start at -spacing*8
            base_start = int((lower - spacing*8) + (z * spacing))
            lungbase[z] = volume[base_start]

        # Images
        image = np.stack([apical, midlung, lungbase], -1)

        # Save the data
        data[index] = {'data': image, 'label': label, 'label_raw': label_raw, 'accno': Accno, 'MRN': MRN, 'file': file}
        index += 1
        pts += 1

        # Garbage
        del volume, mask, image, apical, midlung, lungbase

        # # TODO: Testing
        # for z in range(3): sdd.display_volume(image[..., z], False)
        # if pts > 3: break

        # Save every 20 patients
        if pts % 40 == 0:
            print('%s patients complete, %s images saved' % (pts, index))
            file_root = ('data/Egs_' + str(pts // 40))
            sdl.save_tfrecords(data, 1, file_root=file_root)
            if pts < 45: sdl.save_dict_filetypes(data[0])
            del data
            data = {}

        # All patients done, print the summary message
    print('%s Patients saved, %s failed[No label, Label out of range, Failed load] %s' % (pts, sum(failures), failures))

    # Now create a final protocol buffer
    print('Creating final protocol buffer')
    if data:
        print('%s patients complete, %s images saved' % (pts, index))
        sdl.save_tfrecords(data, 1, file_root='data/Egs_Fin')

    # plt.show()


# Load the protobuf
def load_protobuf(filenames, training=True):

    """
    Loads the protocol buffer into a form to send to shuffle
    """

    # Create a dataset from the protobuf
    dataset = tf.data.TFRecordDataset(filenames)

    _records_call = lambda dataset: \
        sdl.load_tfrecords(dataset, [10, FLAGS.box_dims, FLAGS.box_dims, 3], tf.int16)

    # Parse the record into tensors
    dataset = dataset.map(_records_call, num_parallel_calls=6)

    # Warp the data set
    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=6)

    # Repeat input indefinitely
    dataset = dataset.repeat()

    # Shuffle the dataset then create a batch
    dataset = dataset.shuffle(buffer_size=100)
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

        # Now normalize
        image = tf.image.per_image_standardization(image)

        # Image augmentation. First calc rotation parameters
        angle = tf.random_uniform([1], -0.35, 0.35)

        # Random rotate
        image = tf.contrib.image.rotate(image, angle)

        # Then randomly flip
        image = tf.image.random_flip_left_right(tf.image.random_flip_up_down(image))

        # Random brightness/contrast
        image = tf.image.random_brightness(image, max_delta=1.5)
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

        # Generate random slices to use
        slice_a = tf.squeeze(tf.random_uniform([1], 0, 10, dtype=tf.int32))
        slice_m = tf.squeeze(tf.random_uniform([1], 0, 10, dtype=tf.int32))
        slice_b = tf.squeeze(tf.random_uniform([1], 0, 10, dtype=tf.int32))

        # Apply the slices
        apex, midlung, base = tf.squeeze(apex[slice_a]), tf.squeeze(midlung[slice_m]), tf.squeeze(base[slice_b])

        # Stack the results on a per channel basis
        image = tf.stack([apex, midlung, base], -1)

        # Now normalize
        image = tf.image.per_image_standardization(image)

       # Center crop
        image = tf.image.central_crop(image, 0.85)

        # Reshape image
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims, 3])

        # Display the images
        tf.summary.image('Apex Val', tf.reshape(image[..., 0], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)
        tf.summary.image('Midlung Val', tf.reshape(image[..., 1], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)
        tf.summary.image('Base Val', tf.reshape(image[..., 2], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 8)

    # Make record image
    record['data'] = image

    return record

# pre_proc_25D()
# filenames = tf.placeholder(tf.string, shape=[None])
# load_protobuf(filenames)

    # # TODO: Testing
    # all_files = sdl.retreive_filelist('tfrecords', False, path='data/')
    # train_files = [x for x in all_files if '_9' not in x]
    # sess = tf.InteractiveSession()
    # batch = iterator.get_next()
    # sess.run(iterator.initializer, feed_dict={filenames: train_files})
    # output = sess.run(batch)
    # test_img = output['data'][4, :, :, :, 1]
    # sdd.display_volume(test_img, plot=True)