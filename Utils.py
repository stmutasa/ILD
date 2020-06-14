"""
Utilities to help us out
"""

import SODTester as SDT
import SODLoader as SDL
import SOD_Display as SDD

import numpy as np
from pathlib import Path
from random import shuffle
import matplotlib.pyplot as plt
from scipy.special import softmax as sm
import csv
import os
import cv2
import random
import pickle
import re

# Define the data directory to use
home_dir = str(Path.home()) + '/Code/Datasets/CT_Chest_ILD/'

sdt = SDT.SODTester(True, False)
sdd = SDD.SOD_Display()
sdl = SDL.SODLoader(home_dir)


def save_gender_age():
    """
    Helper to save the gender, age, and accno in a dictionary
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
            print('Unable to load: ', file, '\n')
            failures[2] += 1
            continue

        # Retreive pt info
        Accno = header['tags'].AccessionNumber
        MRN = int(header['tags'].PatientID)
        Age = int(header['tags'].PatientAge[:-1])
        Sex = header['tags'].PatientSex
        try:
            label_raw = int(labels[Accno]['Label'])
        except:
            failures[0] += 1
            del volume
            continue

        # Fix labels per Hiram
        if label_raw < 2:
            label = 1
        elif label_raw == 3:
            label = 0
        else:
            failures[1] += 1
            del volume
            continue

        Accno = int(Accno)

        data[index] = {'Acc': Accno, 'MRN': MRN, 'Age': Age, 'Sex': Sex, 'Raw Label': label_raw, 'Network Label': label}
        index += 1

    # Save the dictionary
    sdl.save_Dict_CSV(data, 'data/Patient_info.csv')


def combine_csvs():
    """
    Combines all the csvs into one .csv
    :return:
    """

    # Load all the dictionaries
    original = sdl.load_CSV_Dict('Acc', 'data/Original.csv')
    network = sdl.load_CSV_Dict('Acc', 'data/Outputs.csv')
    demographics = sdl.load_CSV_Dict('Acc', 'data/Patient_info.csv')

    # Convert all dictionary keys to integers
    original = {int(k): dic for k, dic in original.items()}
    demographics = {int(k): dic for k, dic in demographics.items()}
    network = {int(k[1:-1]): dic for k, dic in network.items()}

    # Initialize
    new_dict, idx = {}, 0

    # Loop through all the original ones
    for index, o_dic in original.items():

        # Initiate demographic and network defaults
        mrn = age = sex = gt = log0 = log1 = wedges = pos_wedges = percent_ILD = smax = 'None'

        # Get the baseline characteristics
        CTPL = o_dic['CTPred_Lancet']
        CTP = o_dic['CTPred']
        PP = o_dic['Path_Raghu']
        CDX = o_dic['Clinical_Dx']
        ER = o_dic['Exclusion_Reason']
        Note = o_dic['Notes']
        try:
            CCf = int(o_dic['Clinical_Confidence'])
        except:
            CCf = 'N/A'

        # Get this patients outputs if they exist
        try:
            # Get the demographic values
            mrn = int(demographics[index]['MRN'])
            age = int(demographics[index]['Age'])
            sex = demographics[index]['Sex']
            gt = int(demographics[index]['Network'])

        except:
            pass

        try:
            # Get the network predictions
            log0 = float(network[index]['Log0'])
            log1 = float(network[index]['Log1'])
            wedges = int(network[index]['Wedges'])
            pos_wedges = int(network[index]['Pos_Wedges'])
            percent_ILD = float(network[index]['PercentILD'][:-1])

            # Calculate softmax
            smax = ('%.3f%%' % (softmax([log0, log1])[gt] * 100))

        except:
            pass

        # Define the dictionary
        new_dict[idx] = {'Accession': index, 'MRN': mrn, 'Age': age, 'Sex': sex, 'Date': o_dic['Date'],
                         'CTPred_Lancet': CTPL, 'CTPred': CTP,
                         'Path_Pred': PP, 'Clinical_Dx': CDX, 'Clinical_Confidence': CCf,
                         'Net_Percent_ILD': percent_ILD,
                         'Net_Confidence': smax, 'Net_Ground_Truth': gt, 'Wedges_Sampled': wedges,
                         'Wedges_Positive': pos_wedges,
                         'Exclusion_Rzn': ER}

        idx += 1

    # Save the new dictionary TODO: We have some kind of writing glitch, save transposed works????
    sdl.save_Dict_CSV(new_dict, 'data/Combined_Outputs.csv', orient='columns')


def softmax(x):
    a = x[0] / (x[0] + x[1])
    b = x[1] / (x[0] + x[1])
    return [a, b]


def save_csvs():
    """
    Combines all the csvs into one .csv
    :return:
    """

    # Load all the dictionaries
    original = sdl.load_CSV_Dict('Acc', 'data/Original.csv')
    network = sdl.load_CSV_Dict('Acc', 'data/Outputs.csv')
    demographics = sdl.load_CSV_Dict('Acc', 'data/Patient_info.csv')

    # Convert all dictionary keys to integers
    original = {int(k): dic for k, dic in original.items()}
    demographics = {int(k): dic for k, dic in demographics.items()}
    network = {int(k[1:-1]): dic for k, dic in network.items()}

    # Save All
    sdl.save_Dict_CSV(original, 'data/original.csv')
    sdl.save_Dict_CSV(demographics, 'data/patient_info.csv')
    sdl.save_Dict_CSV(network, 'data/outputs.csv')


def make_gifs():

    """
    Saves gifs of the overlaid masks and actual volumes
    """

    filenames = sdl.retreive_filelist('nii.gz', include_subfolders=True, path='data/Viz')
    filenames = [x for x in filenames if 'vol' in x]

    for file in filenames:

        # Load volume and mask
        volume = sdl.load_NIFTY(file)
        mask_file = file.replace('vol', 'mask')
        mask = sdl.load_NIFTY(mask_file)

        # Swap axes, currently its upside-down saggital
        volume, mask = np.swapaxes(volume, 0, 1), np.swapaxes(mask, 0, 1)
        volume, mask = np.swapaxes(volume, 1, 2), np.swapaxes(mask, 1, 2)

        # Save gif of volume
        vol_save = 'data/Viz/gifs/' + os.path.basename(file).replace('.nii.gz', '')
        sdl.save_gif_volume(volume, vol_save, swapaxes=False)

        # Save overlaid gif
        overlay_gif = vol_save + '_overlay'
        overlay = sdd.display_whole_vol_overlay(volume, mask, plot=True, ret=True)
        sdl.save_gif_volume(overlay, overlay_gif)

        # Now save prediction gif
        pred_gif = vol_save + '_pred.gif'
        sdt.plot_img_and_mask3D(volume, mask, mask, pred_gif)


def Viz_heatmap():

    """
    Vizualize the generated heatmaps
    """

    filenames = sdl.retreive_filelist('nii.gz', include_subfolders=True, path='data/Viz')
    filenames = [x for x in filenames if 'vol' in x]
    random.shuffle(filenames)

    for file in filenames:

        # Load volumes
        volume = sdl.load_NIFTY(file)
        mask_file = file.replace('vol', 'mask')
        mask = sdl.load_NIFTY(mask_file)
        heat_file = file.replace('vol', 'preds')
        # heat_file = file.replace('vol', 'preds_full')
        heatmap = sdl.load_NIFTY(heat_file)
        regen_file = file.replace('vol', 'box')
        regen_vol = sdl.load_NIFTY(regen_file)

        # Swap axes, currently its upside-down saggital
        volume, mask = np.swapaxes(volume, 0, 1), np.swapaxes(mask, 0, 1)
        heatmap, regen_vol = np.swapaxes(heatmap, 0, 1), np.swapaxes(regen_vol, 0, 1)
        volume, mask = np.swapaxes(volume, 1, 2), np.swapaxes(mask, 1, 2),
        heatmap, regen_vol = np.swapaxes(heatmap, 1, 2), np.swapaxes(regen_vol, 1, 2)

        # Normalize volume for heatmap overlay, we don't want negative numbers!!
        volume = volume.astype(np.float32)
        volume += 1500
        volume /= 1500

        # Generate overlay
        overlay = []
        for z in range(volume.shape[0]):
            overlay.append(sdd.return_heatmap_overlay(volume[z], heatmap[z], threshold=0.00001))
        overlay = np.asarray(overlay)

        # Save file names
        volume_gif_savefile = 'data/Viz/gifs/' + os.path.basename(file).replace('.nii.gz', '')
        # overlay_gif_savefile = volume_gif_savefile + '_FullOverlay'
        # overlay_vol_savefile = file.replace('vol', 'FullOverlay')
        overlay_gif_savefile = volume_gif_savefile + '_Overlay2'
        overlay_vol_savefile = file.replace('vol', 'Overlay2')

        # Save the gifs and volumes
        # sdl.save_gif_volume(volume, volume_gif_savefile, swapaxes=False)
        sdl.save_gif_volume(overlay, overlay_gif_savefile, norm=False)
        sdl.save_volume(overlay, overlay_vol_savefile)

    plt.show()


def Make_graphs():

    """
    Make a myriad of graphs for display:
    Boxes made next to input vol		Half stride	dont matta
    Lung mask next to input			dont matta
    256 = 8
    """

    filenames = sdl.retreive_filelist('nii.gz', include_subfolders=True, path='data/Viz/')
    filenames = [x for x in filenames if 'vol' in x]
    index = 0

    for file in filenames:

        # # Skip undesired files
        accno = int(file.split('/')[-1].split('_')[0])
        # if accno not in des: continue

        # Load volumes: Fulls for comparisons, halfs for preproc graphs
        original = sdl.load_NIFTY(file)
        mask_file = file.replace('vol', 'mask')
        mask = sdl.load_NIFTY(mask_file)
        heatmap_file = file.replace('vol', 'preds_full')
        try:
            heatmap = sdl.load_NIFTY(heatmap_file)
        except:
            continue
        box_file = file.replace('vol', 'box')
        boxes = sdl.load_NIFTY(box_file)

        # Swap axes, currently its upside-down saggital
        original, mask, heatmap, boxes = np.swapaxes(original, 0, 1), np.swapaxes(mask, 0, 1), np.swapaxes(heatmap, 0,
                                                                                                           1), np.swapaxes(
            boxes, 0, 1)
        original, mask, heatmap, boxes = np.swapaxes(original, 1, 2), np.swapaxes(mask, 1, 2), np.swapaxes(heatmap, 1,
                                                                                                           2), np.swapaxes(
            boxes, 1, 2)

        # Normalize volume for heatmap heatmap, we don't want negative numbers!!
        volume = original.astype(np.float32)
        volume += 1500
        volume /= 1500

        # Generate heatmap
        overlay = []
        try:
            for z in range(volume.shape[0]):
                overlay.append(sdd.return_heatmap_overlay(volume[z], heatmap[z], threshold=0.00001))
            overlay = np.asarray(overlay)
        except:
            print('Failed: ', file)
            del original, mask, heatmap, boxes, overlay
            continue

        # Display for us cuz
        print(accno, file)
        # sdd.display_volume(original)
        # sdd.display_volume(mask)
        # sdd.display_volume(boxes)
        sdd.display_volume(overlay)
        del original, mask, heatmap, boxes, overlay
        index += 1
        if index % 25 == 0: plt.show()

    plt.show()


def save_val_vols():

    """
    Helper function to sort the raw downloaded dicoms into a folder structure that makes sense
    MRN/Accno/Series_Time.nii.gz
    MRN/Accno/Series_Time.gif
    MRN/Accno/Series_Time.json
    :return:
    """

    # First retreive the filenames
    path = home_dir + 'Val/'
    folders = sdl.retreive_filelist('*', path=path, include_subfolders=True)
    shuffle(folders)

    # Variables to save
    patient, study = 0, 0

    # Load the images and filter them
    for folder in folders:

        try:
            vols = sdl.load_DICOM_VOLS(folder)
            key = next(iter(vols))
            header = vols[key]['header']
        except Exception as e:
            print('Image Error: %s,  --- Folder: %s' % (e, folder))
            continue

        try:
            Accno = header['tags'].AccessionNumber
            MRN = header['tags'].PatientID
        except Exception as e:
            # Print error then make a dummy header
            print('Header Error: %s,  --- Folder: %s' % (e, folder))
            Accno = folder.split('/')[-2]
            continue

        """
         TODO: Sort the folders and save as niftis
        """

        for series, dict in vols.items():

            # Convert to numpy
            volume = dict['volume']
            volume = np.asarray(volume, np.int16)
            # spacing = str(volume.shape[0]) + '-' + str(dict['spacing'][0])
            spacing = str(dict['spacing']).replace(' ', '-')
            spacing = re.sub('\-\-+', '-', spacing)

            # Skip obvious scout and other small series
            if volume.shape[0] <= 20: continue

            # Get savefile names
            series = series.replace('(', '').replace(')', '').replace('/', '')
            save_root = path.replace('Val', 'Ann')
            fname_vol = ('%s%s_%s_%s_%s.nii.gz' % (save_root, Accno, MRN, series, spacing))

            # Create the root folder
            if not os.path.exists(os.path.dirname(fname_vol)): os.mkdir(os.path.dirname(fname_vol))

            # Save the gif and volume
            print('Saving: ', os.path.basename(fname_vol))

            try:
                sdl.save_volume(volume, fname_vol, compress=True)
                # with open(fname_vol, 'wb') as fp:
                #     pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('\nSaving Error %s: %s,  --- Folder: %s' % (volume.shape, e, folder))
                continue

            # Increment
            study += 1
            del volume

        # Garbage
        patient += 1
        del vols


def save_val_vols_PhillipsDL():
    """
    Helper function to sort the raw PHILLIPS PACS downloaded dicoms into a folder structure that makes sense
    MRN/Accno/Series_Time.nii.gz
    MRN/Accno/Series_Time.gif
    MRN/Accno/Series_Time.json
    :return:
    """

    # Path for this function
    path = home_dir + 'Val_Phillips/'

    # First retreive lists of the the filenames
    folders = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        folders += [os.path.join(dirpath, dir) for dir in dirnames]
    folders = [x for x in folders if 'OBJ_0' in x]

    # Variables to save
    patient, study = 0, 0

    # Load the images and filter them
    for folder in folders:

        # Load the DICOMs
        try:
            volume, header = sdl.load_DICOM_3D(folder, return_header=True)
        except Exception as e:
            print('Image Error: %s,  --- Folder: %s' % (e, folder))
            continue

        try:
            Accno = header['tags'].AccessionNumber
            MRN = header['tags'].PatientID
        except Exception as e:
            print('Header Error: %s,  --- Folder: %s' % (e, folder))
            continue

        """
         TODO: Sort the folders and save as niftis
        """

        # Convert to numpy
        spacing = str(header['spacing']).replace(' ', '-')
        spacing = re.sub('\-\-+', '-', spacing)
        series = header['tags'].SeriesDescription

        # Skip obvious scout and other small series
        if volume.shape[0] <= 20: continue

        # Get savefile names
        series = series.replace('(', '').replace(')', '').replace('/', '')
        save_root = path.replace('Val_Phillips', 'Ann')
        fname_vol = ('%s%s_%s_%s_%s.nii.gz' % (save_root, Accno, MRN, series, spacing))

        # Create the root folder
        if not os.path.exists(os.path.dirname(fname_vol)): os.mkdir(os.path.dirname(fname_vol))

        # Save the gif and volume
        print('Saving: ', os.path.basename(fname_vol))
        save_dict = {'header': header['tags'], 'volume': volume, 'spacing': header['spacing']}

        try:
            sdl.save_volume(volume, fname_vol, compress=True)
            # with open(fname_vol, 'wb') as fp:
            #     pickle.dump(save_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('\nSaving Error %s: %s,  --- Folder: %s' % (volume.shape, e, folder))
            continue

        # Increment
        study += 1
        patient += 1
        del volume


# Viz_heatmap()
Make_graphs()
# save_val_vols()
# save_val_vols_PhillipsDL()
