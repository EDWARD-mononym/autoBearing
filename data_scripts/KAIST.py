import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
import zipfile

from utils import download_file, sliding_window_subsample, normalise_tensor

################################################# DATASET DESCRIPTION ############################################################################
# https://data.mendeley.com/datasets/vxkj334rzv/7
#################################################################################################################################################

class KAIST():
    def __init__(self, args) -> None:
        self.raw_zip_link = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/vxkj334rzv-7.zip'
        self.download_path = Path(f'{args.raw_dir}/KAIST')
        self.processed_path = Path(f'{args.processed_dir}/KAIST')

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
        if not os.path.exists(os.path.join(self.download_path, 'vxkj334rzv-7.zip')):
            print('Downloading KAIST raw files')
            download_file(self.raw_zip_link, self.download_path)

        if not os.path.exists(os.path.join(self.download_path, 'Vibration and Motor Current Dataset of Rolling Element Bearing Under Varying Speed Conditions for Fault Diagnosis Subset1')):  
            with zipfile.ZipFile(os.path.join(self.download_path, 'vxkj334rzv-7.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.download_path)

        if not os.path.exists(os.path.join(self.download_path, 'vibration_ball_0.csv')):
            with zipfile.ZipFile(os.path.join(self.download_path, 'Vibration and Motor Current Dataset of Rolling Element Bearing Under Varying Speed Conditions for Fault Diagnosis Subset1', 'part1.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.download_path)

    def process_data(self):
        self.download_data()

        all_signal, all_label = [], []

        for label, fault_type in enumerate(['normal', 'ball', 'inner', 'outer']):
            fault_signal = []
            for i in range(3):
                file_name = f'vibration_{fault_type}_{i}.csv'
                signal = self.read_csv(file_name) # Should be in the shape (1, 4, 7680000)
                fault_signal.append(signal)
                all_label.append(label)
            fault_signal = torch.concatenate(fault_signal) # Should be in the shape (3, 4, 7680000)
            all_signal.append(fault_signal)
        all_signal = torch.concatenate(all_signal)
        all_label = torch.tensor(all_label)

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
        train_x, train_y = sliding_window_subsample(train_signals, all_label, self.window_size, self.step)
        val_x, val_y = sliding_window_subsample(val_signals, all_label, self.window_size, self.step)
        test_x, test_y = sliding_window_subsample(test_signals, all_label, self.window_size, self.step)

        train = {"samples": train_x, "labels": train_y}
        val = {"samples": val_x, "labels": val_y}
        test = {"samples": test_x, "labels": test_y}

        if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path)
        torch.save(train, os.path.join(self.processed_path, "train.pt"))
        torch.save(val, os.path.join(self.processed_path, "val.pt"))
        torch.save(test, os.path.join(self.processed_path, "test.pt"))

    def read_csv(self, csv_file_name):
        signal = pd.read_csv(os.path.join(self.download_path, csv_file_name)) # Should be in the shape (7680000, 4)
        tensor_signal = torch.tensor(np.array(signal)).T.unsqueeze(0) # Should be in the shape (1, 4, 7680000)
        return tensor_signal