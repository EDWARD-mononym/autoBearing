import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import torch
import zipfile

from utils import download_file, sliding_window_subsample, normalise_tensor, subsample_fewshots

################################################# DATASET DESCRIPTION ############################################################################
# https://github.com/hitwzc/Bearing-datasets#:~:text=HIT%2DSM%20bearing%20datasets%20are,Institute%20of%20Technology%2C%20Harbin%2C%20PR
#################################################################################################################################################

class HITSM():
    def __init__(self, args) -> None:
        self.raw_zip_link = 'https://github.com/hitwzc/Bearing-datasets/archive/refs/heads/main.zip'
        self.download_path = Path(f'{args.raw_dir}/HITSM')
        self.processed_path = Path(f'{args.processed_dir}')

        self.window_size = args.window_size
        self.stride = args.stride
        self.step = int(self.window_size * self.stride)
        self.few_shots = args.few_shots

        self.train_size = args.train_size
        self.test_size = args.test_size
        self.val_size = args.val_size

        self.fttp = args.fttp
        self.normalise = args.normalise

    def download_data(self):
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        #* ############################################################################################
        #* Download raw zip file if it hasn't already been downloaded yet
        #* ############################################################################################
        if not os.path.exists(os.path.join(self.download_path, 'main.zip')):
            print('Downloading HITSM raw files')
            download_file(self.raw_zip_link, self.download_path)

        if not os.path.exists(os.path.join(self.download_path, 'Bearing-datasets-main')):  
            with zipfile.ZipFile(os.path.join(self.download_path, 'main.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.download_path)

    def process_data(self):
        self.download_data()

        #! Self-built-dataset
        if not os.path.exists(os.path.join(self.processed_path, 'HITSM_self_built', 'train.pt')):
            class_dict = {
                0: ['Normal_600.mat', 'Normal_900.mat', 'Normal_1200.mat'],
                1: ['IR2_600.mat', 'IR2_900.mat', 'IR2_1200.mat'],
                2: ['IR5_600.mat', 'IR5_900.mat', 'IR5_1200.mat'],
                3: ['IR8_600.mat', 'IR8_900.mat', 'IR8_1200.mat'],
                4: ['OR2_600.mat', 'OR2_900.mat', 'OR2_1200.mat'],
                5: ['OR5_600.mat', 'OR5_900.mat', 'OR5_1200.mat'],
                6: ['OR8_600.mat', 'OR8_900.mat', 'OR8_1200.mat']
            }

            labels, all_signal = [], []
            for class_label, file_list in class_dict.items():
                for file in file_list:
                    file_path = os.path.join(self.download_path, 'Bearing-datasets-main', 'Self-built dataset', file)
                    signal = self.read_mat_file(file_path)
                    all_signal.append(signal)
                    labels.append(class_label)
            
            all_signal = torch.cat(all_signal, 0)
            labels = torch.tensor(labels)

            if self.normalise: #* If true, normalise the signal so that it has mean=0 and s.d=1
                all_signal = normalise_tensor(all_signal)
            
            # Split the signal into train and testing
            signal_length = all_signal.shape[2]
            train_size = int(signal_length * self.train_size)
            test_size = int(signal_length * self.test_size)
            val_size = signal_length - train_size - test_size

            train_signals = all_signal[:,:,:train_size]
            val_signals = all_signal[:,:,train_size:train_size+val_size]
            test_signals = all_signal[:,:,train_size+val_size:]

            # # Subsample with sliding window
            train_x, train_y = sliding_window_subsample(train_signals, labels, self.window_size, self.step)
            val_x, val_y = sliding_window_subsample(val_signals, labels, self.window_size, self.step)
            test_x, test_y = sliding_window_subsample(test_signals, labels, self.window_size, self.step)

            train = {"samples": train_x, "labels": train_y}
            val = {"samples": val_x, "labels": val_y}
            test = {"samples": test_x, "labels": test_y}

            if not os.path.exists(os.path.join(self.processed_path, 'HITSM_self_built')):
                os.makedirs(os.path.join(self.processed_path, 'HITSM_self_built'))
            torch.save(train, os.path.join(self.processed_path, 'HITSM_self_built', "train.pt"))
            torch.save(val, os.path.join(self.processed_path, 'HITSM_self_built', "val.pt"))
            torch.save(test, os.path.join(self.processed_path, 'HITSM_self_built', "test.pt"))

        #! SpectraQuest dataset
        if not os.path.exists(os.path.join(self.processed_path, 'HITSM_SpectraQuest', 'train.pt')):
            class_dict = {
                0: ['Normal_600.mat', 'Normal_900.mat', 'Normal_1200.mat'],
                1: ['IR2_600.mat', 'IR2_900.mat', 'IR2_1200.mat'],
                2: ['IR5_600.mat', 'IR5_900.mat', 'IR5_1200.mat'],
                3: ['IR8_600.mat', 'IR8_900.mat', 'IR8_1200.mat'],
                4: ['OR2_600.mat', 'OR2_900.mat', 'OR2_1200.mat'],
                5: ['OR5_600.mat', 'OR5_900.mat', 'OR5_1200.mat'],
                6: ['OR8_600.mat', 'OR8_900.mat', 'OR8_1200.mat']
            }

            labels, all_signal = [], []
            for class_label, file_list in class_dict.items():
                for file in file_list:
                    file_path = os.path.join(self.download_path, 'Bearing-datasets-main', 'SpectraQuest MFS dataset', file)
                    signal = self.read_mat_file(file_path)
                    all_signal.append(signal)
                    labels.append(class_label)
            
            all_signal = torch.cat(all_signal, 0)
            labels = torch.tensor(labels)

            if self.normalise: #* If true, normalise the signal so that it has mean=0 and s.d=1
                all_signal = normalise_tensor(all_signal)
            
            # Split the signal into train and testing
            signal_length = all_signal.shape[2]
            train_size = int(signal_length * self.train_size)
            test_size = int(signal_length * self.test_size)
            val_size = signal_length - train_size - test_size

            train_signals = all_signal[:,:,:train_size]
            val_signals = all_signal[:,:,train_size:train_size+val_size]
            test_signals = all_signal[:,:,train_size+val_size:]

            # # Subsample with sliding window
            train_x, train_y = sliding_window_subsample(train_signals, labels, self.window_size, self.step)
            val_x, val_y = sliding_window_subsample(val_signals, labels, self.window_size, self.step)
            test_x, test_y = sliding_window_subsample(test_signals, labels, self.window_size, self.step)

            train = {"samples": train_x, "labels": train_y}
            val = {"samples": val_x, "labels": val_y}
            test = {"samples": test_x, "labels": test_y}

            if not os.path.exists(os.path.join(self.processed_path, 'HITSM_SpectraQuest')):
                os.makedirs(os.path.join(self.processed_path, 'HITSM_SpectraQuest'))
            torch.save(train, os.path.join(self.processed_path, 'HITSM_SpectraQuest', "train.pt"))
            torch.save(val, os.path.join(self.processed_path, 'HITSM_SpectraQuest', "val.pt"))
            torch.save(test, os.path.join(self.processed_path, 'HITSM_SpectraQuest', "test.pt"))


    def read_mat_file(self, file_path):
        mat_file = loadmat(file_path)
        key = self.get_mat_key(mat_file)
        signal = torch.tensor(mat_file[key]) # Should return a (262144, 1)

        return signal.T.unsqueeze(0) # Convert to (1, 1, 262144)

    def get_mat_key(self, mat_file):
        keys = []
        for key in mat_file:
            keys.append(key)
        return keys[-1]