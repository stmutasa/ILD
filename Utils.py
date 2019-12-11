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

home_dir = str(Path.home()) + '/PycharmProjects/Datasets/CT_Chest_ILD/'

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


# save_gender_age()
# combine_csvs()

save_csvs()
