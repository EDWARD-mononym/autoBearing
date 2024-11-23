from bs4 import BeautifulSoup
import numpy as np
import os
import pandas as pd
from pathlib import Path
import patoolib
import requests
import torch
from scipy.io import loadmat
import zipfile

from utils import download_file, set_seed_and_deterministic, train_test_split, sliding_window_subsample, normalise_tensor, subsample_fewshots

#? Dataset description: https://data.mendeley.com/datasets/v43hmbwxpm/1

class UO():
    def __init__(self, args) -> None:
        self.raw_zip_link = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/v43hmbwxpm-1.zip'
        self.download_path = Path(f'{args.raw_dir}/UO')
        self.processed_path = Path(f'{args.processed_dir}/UO')

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
        if not os.path.exists(os.path.join(self.download_path, 'v43hmbwxpm-1.zip')):
            print('Downloading UO raw files')
            download_file(self.raw_zip_link, self.download_path)
            
        if not os.path.exists(os.path.join(self.download_path, 'H-A-1.mat')):  
            with zipfile.ZipFile(os.path.join(self.download_path, 'v43hmbwxpm-1.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.download_path)

    def process_data(self):
        self.download_data() # Download PU raw data if it doesn't exist in self.download_path

        if not os.path.exists(os.path.join(self.processed_path)):
            healthy = ['H-A-1.mat', 'H-A-2.mat', 'H-A-3.mat', 'H-B-1.mat', 'H-B-2.mat', 'H-B-3.mat', 'H-C-1.mat', 'H-C-2.mat', 'H-C-3.mat', 'H-D-1.mat', 'H-D-2.mat', 'H-D-3.mat']
            inner = ['I-A-1.mat', 'I-A-2.mat', 'I-A-3.mat', 'I-B-1.mat', 'I-B-2.mat', 'I-B-3.mat', 'I-C-1.mat', 'I-C-2.mat', 'I-C-3.mat', 'I-D-1.mat', 'I-D-2.mat', 'I-D-3.mat']
            outer = ['O-A-1.mat', 'O-A-2.mat', 'O-A-3.mat', 'O-B-1.mat', 'O-B-2.mat', 'O-B-3.mat', 'O-C-1.mat', 'O-C-2.mat', 'O-C-3.mat', 'O-D-1.mat', 'O-D-2.mat', 'O-D-3.mat']

            healthy_signals = self.read_list_files(healthy)
            inner_signals = self.read_list_files(inner)
            outer_signals = self.read_list_files(outer)

            # Split healthy signals into train, val and test
            set_seed_and_deterministic(42)
            train_healthy, test_val_healthy = train_test_split(self.train_size, healthy_signals)
            relative_healthy_test_size = self.test_size / (self.test_size + self.val_size)
            test_healthy, val_healthy = train_test_split(relative_healthy_test_size, test_val_healthy)

            # Split inner signals into train, val and test
            train_inner, test_val_inner = train_test_split(self.train_size, inner_signals)
            relative_inner_test_size = self.test_size / (self.test_size + self.val_size)
            test_inner, val_inner = train_test_split(relative_inner_test_size, test_val_inner)

            # Split outer signals into train, val and test
            train_outer, test_val_outer = train_test_split(self.train_size, outer_signals)
            relative_outer_test_size = self.test_size / (self.test_size + self.val_size)
            test_outer, val_outer = train_test_split(relative_outer_test_size, test_val_outer)

            # Combine healthy and faulty samples
            healthy_train_label = torch.full([len(train_healthy)], 0)
            inner_train_label = torch.full([len(train_inner)], 1)
            outer_train_label = torch.full([len(train_outer)], 2)
            train_x = torch.cat((train_healthy, train_inner, train_outer), 0)
            train_y = torch.cat((healthy_train_label, inner_train_label, outer_train_label), 0)

            healthy_val_label = torch.full([len(val_healthy)], 0)
            inner_val_label = torch.full([len(val_inner)], 1)
            outer_val_label = torch.full([len(val_outer)], 2)
            val_x = torch.cat((val_healthy, val_inner, val_outer), 0)
            val_y = torch.cat((healthy_val_label, inner_val_label, outer_val_label), 0)

            healthy_test_label = torch.full([len(test_healthy)], 0)
            inner_test_label = torch.full([len(test_inner)], 1)
            outer_test_label = torch.full([len(test_outer)], 2)
            test_x = torch.cat((test_healthy, test_inner, test_outer), 0)
            test_y = torch.cat((healthy_test_label, inner_test_label, outer_test_label), 0)

            # Subsample with sliding window
            train_x, train_y = sliding_window_subsample(train_x, train_y, self.window_size, self.step)
            val_x, val_y = sliding_window_subsample(val_x, val_y, self.window_size, self.step)
            test_x, test_y = sliding_window_subsample(test_x, test_y, self.window_size, self.step)

            # Save data
            train = {"samples": train_x, "labels": train_y}
            val = {"samples": val_x, "labels": val_y}
            test = {"samples": test_x, "labels": test_y}
            if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path)
            torch.save(train, os.path.join(self.processed_path, "train.pt"))
            torch.save(val, os.path.join(self.processed_path, "val.pt"))
            torch.save(test, os.path.join(self.processed_path, "test.pt"))

    def read_list_files(self, list_of_file):
        signals = []
        for file_name in list_of_file:
            path = os.path.join(self.download_path, file_name)
            single_signal = self.load_mat(path)
            signals.append(single_signal)

        # Combine all the signals
        signals = torch.stack(signals, dim=0) # Should be in the shape torch.Size([13, 1, 2000000])
        return signals


    def load_mat(self, mat_file_path):
        mat_file = loadmat(mat_file_path)
        channel_1 = torch.tensor(mat_file['Channel_1']) # Should be in the shape torch.Size([2000000, 1])
        # channel_2 = mat_file['Channel_2'] # Not used for now

        channel_1 = channel_1.t() # Making into the shape torch.Size([ 1, 2000000])
        return channel_1