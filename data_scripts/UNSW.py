import numpy as np
import os
import pandas as pd
from pathlib import Path
import py7zr
import torch
from scipy.io import loadmat
import zipfile

from utils import download_file, sliding_window_subsample, normalise_tensor, subsample_fewshots

################################################# DATASET DESCRIPTION ############################################################################
# https://data.mendeley.com/datasets/h4df4mgrfb/3
#################################################################################################################################################

class UNSW():
    def __init__(self, args) -> None:
        self.raw_zip_link = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/h4df4mgrfb-3.zip'
        self.download_path = Path(f'{args.raw_dir}/UNSW')
        self.processed_path = Path(f'{args.processed_dir}/UNSW')

        self.window_size = args.window_size
        self.stride = args.stride
        self.step = int(self.window_size * self.stride)
        self.few_shots = args.few_shots

        self.fttp = args.fttp
        self.normalise = args.normalise

    def download_data(self):
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        #* ############################################################################################
        #* Download raw zip file if it hasn't already been downloaded yet
        #* ############################################################################################
        if not os.path.exists(os.path.join(self.download_path, 'h4df4mgrfb-3.zip')):
            print('Downloading UNSW raw files')
            download_file(self.raw_zip_link, self.download_path)

        if not os.path.exists(os.path.join(self.download_path, 'h4df4mgrfb-3')):  
            with zipfile.ZipFile(os.path.join(self.download_path, 'h4df4mgrfb-3.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.download_path)

        if not os.path.exists(os.path.join(self.download_path, 'Test 1')):
            with py7zr.SevenZipFile(os.path.join(self.download_path, 'h4df4mgrfb-3', 'Test 1.7z'), mode='r') as archive:
                archive.extractall(path=self.download_path)
            with py7zr.SevenZipFile(os.path.join(self.download_path, 'h4df4mgrfb-3', 'Test 2.7z'), mode='r') as archive:
                archive.extractall(path=self.download_path)
            with py7zr.SevenZipFile(os.path.join(self.download_path, 'h4df4mgrfb-3', 'Test 3.7z'), mode='r') as archive:
                archive.extractall(path=self.download_path)
            with py7zr.SevenZipFile(os.path.join(self.download_path, 'h4df4mgrfb-3', 'Test 4.7z'), mode='r') as archive:
                archive.extractall(path=self.download_path)

    def process_data(self):
        self.download_data()

        if not os.path.exists(os.path.join(self.processed_path, 'train.pt')):
            test_1 = self.read_signal('Test 1')
            test_2 = self.read_signal('Test 2')
            test_3 = self.read_signal('Test 3')
            test_4 = self.read_signal('Test 4')

            if self.normalise: #* If true, normalise the signal so that it has mean=0 and s.d=1
                test_1 = normalise_tensor(test_1)
                test_2 = normalise_tensor(test_2)
                test_3 = normalise_tensor(test_3)
                test_4 = normalise_tensor(test_4)

            #* Create RUL by counting how many timestep a snapshot has until the last snapshot
            test_1_labels = torch.tensor(np.arange(len(test_1)-1, -1, -1))
            test_2_labels = torch.tensor(np.arange(len(test_2)-1, -1, -1))
            test_3_labels = torch.tensor(np.arange(len(test_3)-1, -1, -1))
            test_4_labels = torch.tensor(np.arange(len(test_4)-1, -1, -1))

            #* Convert RUL from timestep unit to % health remaining
            test_1_labels = test_1_labels / (len(test_1_labels)-1)
            test_2_labels = test_2_labels / (len(test_2_labels)-1)
            test_3_labels = test_3_labels / (len(test_3_labels)-1)
            test_4_labels = test_4_labels / (len(test_4_labels)-1)

            #* Subsample the signal such that it has a length of self.window_size
            test_1, test_1_labels = sliding_window_subsample(test_1, test_1_labels, self.window_size, self.step)
            test_2, test_2_labels = sliding_window_subsample(test_2, test_2_labels, self.window_size, self.step)
            test_3, test_3_labels = sliding_window_subsample(test_3, test_3_labels, self.window_size, self.step)
            test_4, test_4_labels = sliding_window_subsample(test_4, test_4_labels, self.window_size, self.step)

            train_signals, train_lables = torch.cat((test_1, test_2), dim=0), torch.cat((test_1_labels, test_2_labels), dim=0)
            val_signals, val_labels = test_3, test_3_labels
            test_signals, test_labels = test_4, test_4_labels

            train = {"samples": train_signals, "labels": train_lables}
            val = {"samples": val_signals, "labels": val_labels}
            test = {"samples": test_signals, "labels": test_labels}

            if not os.path.exists(self.processed_path):
                    os.makedirs(self.processed_path)
            torch.save(train, os.path.join(self.processed_path, "train.pt"))
            torch.save(val, os.path.join(self.processed_path, "val.pt"))
            torch.save(test, os.path.join(self.processed_path, "test.pt"))

    def read_signal(self, test_name):
        horizontal_list, vertical_list = [], []
        for file_name in os.listdir(os.path.join(self.download_path, test_name, '6Hz')):
            horizontal, vertical = self.get_x_y(test_name, file_name)
            horizontal_list.append(horizontal)
            vertical_list.append(vertical)
            
        # Convert the list into a np array
        horizontal_signal, vertical_signal = np.vstack(horizontal_list), np.vstack(vertical_list)
        horizontal_tensor, vertical_tensor = torch.tensor(horizontal_signal).unsqueeze(1), torch.tensor(vertical_signal).unsqueeze(1)
        
        return torch.concatenate((horizontal_tensor, vertical_tensor), axis=1)

    def get_x_y(self, test_name, file_name):
        mat_file = loadmat(os.path.join(self.download_path,  test_name, '6Hz', file_name))
        horizontal_signal = mat_file['accH'].T #? Should be in the shape (1, 614400)
        vertical_signal = mat_file['accV'].T
        return horizontal_signal, vertical_signal