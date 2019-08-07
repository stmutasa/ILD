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
    data, track = {}, {}
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
        display.append(Accno)
        index += 1
        pts += 1

        # Garbage
        del volume, mask, image, apical, midlung, lungbase

        # Save every 20 patients
        if pts % 40 == 0:
            print('%s patients complete, %s images saved' % (pts, index))
            print ('Patients in this protobuf: \n%s' %display)
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


def create_test_set(dims=512):

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
    data = [dict() for x in range(6)]
    display, failures = [], [0, 0, 0]

    # The testing splits for SKYNET
    test_split = {0: ['2622480', '2579399', '2656231', '1672867', '2017547', '1448231', '675864', '2650642', '1650485',
                      '362224', '2226697', '2546442', '2606334', '2541228', '2614734', '933138', '2465890', '2306373',
                      '1364924', '2665186', '1873087', '2058401', '2778324', '1882931', '1826770', '1969397', '2547471',
                      '1664401', '1288811', '2440929', '627947', '2164344', '2389971', '1675180', '2528945', '2072042',
                      '3067337', '2577915', '2656226', '3210138'],
                  1: ['1730867', '2644468', '1422664', '2918675', '2637791', '2863479', '2563336', '730120', '1742979',
                      '2800070', '1679995', '2474151', '2062061', '2775371', '2586787', '1232132', '2838409', '2492366',
                      '1202999', '2180495', '1615657', '2720361', '2275584', '2392471', '2840663', '2492689', '2683227',
                      '2312643', '2499683', '2419194', '1808842', '1099623', '1800303', '2044113', '2655723', '1215476',
                      '3217072','2251829', '2879875', '2956067'],
                  2: ['2879524', '2622637', '2119303', '2991439', '2650242', '3430200', '2076072', '2891181', '1603947',
                      '888448', '3228459', '2109344', '2264605', '2857282', '1889463', '2809579', '2930847', '2699865',
                      '2711117', '617609', '3045880', '2702571', '2381358', '2432325', '1258236', '1558166', '3191026',
                      '2899782', '2615652', '2669826', '2376535', '1918805', '1389305', '1644274', '1896504', '1424094',
                      '2713452', '2318793', '1315965', '1527297'],
                  3: ['2382954', '2834754', '1964271', '1846247', '2736072', '2355595', '2566574', '2715192', '2727520',
                      '647955', '2831984', '2316235', '2563998', '906611', '2688266', '1605810', '1707445', '2735605',
                      '1074859', '1749212', '2702610', '1082751', '2567796', '1367998', '2561392', '1533966', '615121',
                      '1402540', '2831875', '2495091', '2320549', '3158609', '707500', '1904605', '2716332', '2079411',
                      '1352011', '2730430', '1273996', '2431046'],
                  4: ['1766597', '2936015', '2052480', '2074951', '1901745', '1112771', '3022266', '1887799', '2579699',
                      '2794336', '2121806', '2880987', '2706717', '2865710', '1213414', '1543788', '3154971', '2324800',
                      '2847125','2935953', '2036053', '2613768', '1345097', '2660555', '2711262', '3030338', '1056882',
                      '775215', '2418372', '2973183', '1450346', '1253829', '2871017', '633010', '1970892', '2825232',
                      '2570178', '2585899', '1409952', '2822515'],
                  5: ['2667870', '2554416', '2946793', '2637525', '1878716', '2982124', '2951671', '2191314', '2673670',
                      '2651645', '642859', '3136738', '2856223', '1713690', '1157062', '1797479', '3010504', '2706805',
                      '2255159', '1534521','2349032']
                  }

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

        # Create the empty arrays we will fill. Testing set will have 5 slices per
        apical = np.zeros(shape=[5, dims, dims], dtype=np.int16)
        midlung, lungbase = np.zeros_like(apical), np.zeros_like(apical)

        # Now save the segments. Loop every 3 slices
        for z in range (5):

            # Apical segment, start at -spacing
            apical_start = int((apex-spacing) + (z*spacing))
            apical[z] = volume[apical_start]

            # Midlung segment, start at -spacing*2.5
            midlung_start = int((mid - spacing*2.5) + (z * spacing))
            midlung[z] = volume[midlung_start]

            # Apical segment, start at -spacing*4
            base_start = int((lower - spacing*4) + (z * spacing))
            lungbase[z] = volume[base_start]


        # Find the correct dictionary to save to
        for key, ddict in test_split.items():
            if Accno in ddict:
                save_dict_num = key
                break

        # Save all the data
        for ap in range (5):
            for ml in range(5):
                for ba in range(5):

                    # Images
                    image = np.stack([apical[ap], midlung[ml], lungbase[ba]], -1)

                    # Dummy filename
                    file_save = 'pt_num_' + str(pts) + '_apex_' + str(ap) + '_mid_' + str(ml) + '_ba_' + str(ba)

                    # Save the example to the correct dictionary
                    data[save_dict_num][index] =  {'data': image, 'label': label, 'label_raw': label_raw, 'accno': Accno, 'MRN': MRN, 'file': file_save}
                    display.append(Accno)
                    index += 1

                    del image

        pts += 1

        # Garbage
        del volume, mask, apical, midlung, lungbase

    # All patients done, print the summary message
    print('%s patients complete, %s TEST images saved, Saving the dictionaries...' % (pts, index))
    print('%s failed: [No label, Label out of range, Failed load] %s' % (sum(failures), failures))

    # Now create a final protocol buffer
    for dd in range (6):
        print('%s Patients and %s examples in this dictionary' % (len(test_split[dd]), len(data[dd])))
        file_root = ('data/Test_' + str(dd+1))
        sdl.save_tfrecords(data, 1, file_root=file_root)


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

        # We need to iterate through every slice
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

       # Center crop
        image = tf.image.central_crop(image, 0.85)

        # Reshape image
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims, 3])

    # Make record image
    record['data'] = image

    return record

# pre_proc_25D()
# filenames = tf.placeholder(tf.string, shape=[None])
# load_protobuf(filenames)