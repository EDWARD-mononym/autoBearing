import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
from scipy.io import loadmat
import zipfile

from utils import download_file, sliding_window_subsample, normalise_tensor, subsample_fewshots

################################################# DATASET DESCRIPTION ############################################################################
# https://www.mfpt.org/fault-data-sets/
#################################################################################################################################################

class MFPT():
    def __init__(self, args) -> None:
        self.raw_zip_link = 'https://www.mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip'
        self.download_path = Path(f'{args.raw_dir}/MFPT')
        self.processed_path = Path(f'{args.processed_dir}/MFPT')

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
        if not os.path.exists(os.path.join(self.download_path, 'MFPT-Fault-Data-Sets-20200227T131140Z-001.zip')):
            print('Downloading MFPT raw files')
            download_file(self.raw_zip_link, self.download_path)

        if not os.path.exists(os.path.join(self.download_path, 'MFPT Fault Data Sets')):  
            with zipfile.ZipFile(os.path.join(self.download_path, 'MFPT-Fault-Data-Sets-20200227T131140Z-001.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.download_path)

    def process_data(self):
        self.download_data()

        train_x, train_y = [], []
        val_x, val_y = [], []
        test_x, test_y = [], []
        for class_label, folder_name in zip([0, 1, 1, 2], ['1 - Three Baseline Conditions', '2 - Three Outer Race Fault Conditions', '3 - Seven More Outer Race Fault Conditions', '4 - Seven Inner Race Fault Conditions']):
            folder_path = os.path.join(self.download_path, 'MFPT Fault Data Sets', folder_name)
            fault_signal = []
            for file_name in os.listdir(folder_path):
                if file_name[-4:] == '.mat':
                    signal = self.read_mat(os.path.join(folder_path, file_name))
                    fault_signal.append(signal)

            # Because each signal has different length, we process them here first
            fault_signal = torch.concatenate(fault_signal) # (B, C, L)

            # Split the signal into train and testing
            signal_length = fault_signal.shape[2]
            train_size = int(signal_length * self.train_size)
            test_size = int(signal_length * self.test_size)
            val_size = signal_length - train_size - test_size

            train_signals = fault_signal[:,:,:train_size]
            val_signals = fault_signal[:,:,train_size:train_size+val_size]
            test_signals = fault_signal[:,:,train_size+val_size:]

            # # Subsample with sliding window
            train_signal, train_label = sliding_window_subsample(train_signals, torch.full([len(train_signals)], class_label), self.window_size, self.step)
            val_signal, val_label = sliding_window_subsample(val_signals, torch.full([len(val_signals)], class_label), self.window_size, self.step)
            test_signal, test_label = sliding_window_subsample(test_signals, torch.full([len(test_signals)], class_label), self.window_size, self.step)

            train_x.append(train_signal)
            train_y.append(train_label)
            val_x.append(val_signal)
            val_y.append(val_label)
            test_x.append(test_signal)
            test_y.append(test_label)

        train_x = torch.concatenate(train_x)
        train_y = torch.concatenate(train_y)
        val_x = torch.concatenate(val_x)
        val_y = torch.concatenate(val_y)
        test_x = torch.concatenate(test_x)
        test_y = torch.concatenate(test_y)

        train = {"samples": train_x, "labels": train_y}
        val = {"samples": val_x, "labels": val_y}
        test = {"samples": test_x, "labels": test_y}

        if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path)
        torch.save(train, os.path.join(self.processed_path, "train.pt"))
        torch.save(val, os.path.join(self.processed_path, "val.pt"))
        torch.save(test, os.path.join(self.processed_path, "test.pt"))

                

    def read_mat(self, mat_file):
        mat_file = loadmat(mat_file)
        index = self.find_right_index(mat_file)
        signal = mat_file['bearing'][0][0][index] # Should be in the shape (L, 1)
        signal = torch.tensor(signal).T.unsqueeze(0) # Should be in the shape (1, 1, L)

        if signal.shape[2] >= 500000: # Need to downsample to match sampling rate
            signal = signal[::2] # Downsample by taking every other signal

        return signal

    def find_right_index(self, mat_file):
        signal = mat_file['bearing'][0][0]
        for index in range(len(signal)):
            if signal[index].shape[0] >= 100:
                return index