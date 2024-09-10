import numpy as np
import os
import pandas as pd
from pathlib import Path
import patoolib
import torch
import zipfile

from utils import set_seed_and_deterministic, download_file

class IMS():
    def __init__(self, args) -> None:
        self.raw_zip_link = 'https://data.nasa.gov/download/brfb-gzcv/application%2Fzip'
        self.download_path = Path(f'{args.raw_dir}/IMS')
        self.processed_path = Path(f'{args.processed_dir}/IMS')

        self.window_size = args.window_size
        self.stride = args.stride
        self.step = int(self.window_size * self.stride)

    def download_data(self):
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        #* ############################################################################################
        #* Download raw zip file if it hasn't already been downloaded yet
        #* ############################################################################################
        if not os.path.exists(os.path.join(self.download_path, 'IMS.zip')):
            print('Downloading IMS raw zip files')
            download_file(self.raw_zip_link, self.download_path, 'IMS.zip')


        if not os.path.exists(os.path.join(self.download_path, 'IMS')):
            with zipfile.ZipFile(os.path.join(self.download_path, 'IMS.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.download_path)

        if not os.path.exists(os.path.join(self.download_path, '1st_test')):
            patoolib.extract_archive(os.path.join(self.download_path, 'IMS', '1st_test.rar'), outdir=self.download_path)
        if not os.path.exists(os.path.join(self.download_path, '2nd_test')):
            patoolib.extract_archive(os.path.join(self.download_path, 'IMS', '2nd_test.rar'), outdir=self.download_path)
        if not os.path.exists(os.path.join(self.download_path, '4th_test')):
            patoolib.extract_archive(os.path.join(self.download_path, 'IMS', '3rd_test.rar'), outdir=self.download_path)

    def process_data(self):
        self.download_data()

        if not os.path.exists(os.path.join(self.processed_path, 'train.pt')):
            _, _, bearing1_3, _ = self.read_1st_experiment()
            bearing2_1, _, _, _ = self.read_2nd_experiment()
            _, _, bearing3_3, _ = self.read_3rd_experiment()

            bearing1_3_x = torch.tensor(bearing1_3[2006:2155]) # Training
            bearing2_1_x = torch.tensor(bearing2_1[700:970]) # Validation
            bearing3_3_x = torch.tensor(bearing3_3[5967:6261]) # Testing

            bearing1_3_y = torch.tensor(np.arange(len(bearing1_3_x)-1, -1, -1))
            bearing2_1_y = torch.tensor(np.arange(len(bearing2_1_x)-1, -1, -1))
            bearing3_3_y = torch.tensor(np.arange(len(bearing3_3_x)-1, -1, -1))

            # Normalising x and y values
            bearing1_3_x = (bearing1_3_x-torch.mean(bearing1_3_x))/torch.std(bearing1_3_x)
            bearing2_1_x = (bearing2_1_x-torch.mean(bearing2_1_x))/torch.std(bearing2_1_x)
            bearing3_3_x = (bearing3_3_x-torch.mean(bearing3_3_x))/torch.std(bearing3_3_x)

            bearing1_3_y = bearing1_3_y / (len(bearing1_3_y)-1)
            bearing2_1_y = bearing2_1_y / (len(bearing2_1_y)-1)
            bearing3_3_y = bearing3_3_y / (len(bearing3_3_y)-1)

            bearing1_3_x, bearing1_3_y = self.sliding_window_subsample(bearing1_3_x, bearing1_3_y)
            bearing2_1_x, bearing2_1_y = self.sliding_window_subsample(bearing2_1_x, bearing2_1_y)
            bearing3_3_x, bearing1_3_y = self.sliding_window_subsample(bearing3_3_x, bearing3_3_y)

            train = {"samples": bearing1_3_x, "labels": bearing1_3_y}
            val = {"samples": bearing2_1_x, "labels": bearing2_1_y}
            test = {"samples": bearing3_3_x, "labels": bearing3_3_y}

            if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path)

            torch.save(train, os.path.join(self.processed_path, "train.pt"))
            torch.save(val, os.path.join(self.processed_path, "val.pt"))
            torch.save(test, os.path.join(self.processed_path, "test.pt"))

    def sliding_window_subsample(self, tensor_x, tensor_y):
        tensor_x = tensor_x.unsqueeze(1).unfold(2, self.window_size, self.step)
        B, C, W, L = tensor_x.size() # Get the tensor dimensions for reshaping
        tensor_x = tensor_x.reshape(B*W, C, L)
        tensor_y.unsqueeze(1).repeat(1, W).reshape(B*W)
        return tensor_x, tensor_y

    def read_1st_experiment(self):
        snap_shot_list = os.listdir(os.path.join(self.download_path, '1st_test'))
        bearing1_1_list, bearing1_2_list, bearing1_3_list, bearing1_4_list = [], [], [], []
        for i, snap_shot in enumerate(snap_shot_list):
            # bearing1_1_snapshot, bearing1_2_snapshot, bearing1_3_snapshot, bearing1_4_snapshot = self.read_1st_file(snap_shot)
            _, _, bearing1_3_snapshot, _ = self.read_1st_file(snap_shot)

            # bearing1_1_list.append(bearing1_1_snapshot)
            # bearing1_2_list.append(bearing1_2_snapshot)
            bearing1_3_list.append(bearing1_3_snapshot)
            # bearing1_4_list.append(bearing1_4_snapshot)
            
        # Convert the list into a np array
        # bearing1_1 = np.vstack(bearing1_1_list)
        # bearing1_2 = np.vstack(bearing1_2_list)
        bearing1_3 = np.vstack(bearing1_3_list)
        # bearing1_4 = np.vstack(bearing1_4_list)
        
        # return bearing1_1, bearing1_2, bearing1_3, bearing1_4
        return None, None, bearing1_3, None

    def read_1st_file(self, file_name):
        file_path = os.path.join(self.download_path, '1st_test', file_name)
        signal_df = pd.read_csv(file_path, sep='\t', header=None)
        
        # Return signal of bearing1_1, bearing1_2, bearing1_3, bearing1_4
        return np.array(signal_df[0]), np.array(signal_df[2]), np.array(signal_df[4]), np.array(signal_df[6])

    def read_2nd_experiment(self):
        snap_shot_list = os.listdir(os.path.join(self.download_path, '2nd_test'))
        bearing2_1_list, bearing2_2_list, bearing2_3_list, bearing2_4_list = [], [], [], []
        for i, snap_shot in enumerate(snap_shot_list):
            # bearing2_1_snapshot, bearing2_2_snapshot, bearing2_3_snapshot, bearing2_4_snapshot = self.read_2nd_file(snap_shot)
            bearing2_1_snapshot, _, _, _ = self.read_2nd_file(snap_shot)

            bearing2_1_list.append(bearing2_1_snapshot)
            # bearing2_2_list.append(bearing2_2_snapshot)
            # bearing2_3_list.append(bearing2_3_snapshot)
            # bearing2_4_list.append(bearing2_4_snapshot)
            
        # Convert the list into a np array
        bearing2_1 = np.vstack(bearing2_1_list)
        # bearing2_2 = np.vstack(bearing2_2_list)
        # bearing2_3 = np.vstack(bearing2_3_list)
        # bearing2_4 = np.vstack(bearing2_4_list)
        
        # return bearing2_1, bearing2_2, bearing2_3, bearing2_4
        return bearing2_1, None, None, None

    def read_2nd_file(self, file_name):
        file_path = os.path.join(self.download_path, '2nd_test', file_name)
        signal_df = pd.read_csv(file_path, sep='\t', header=None)
        # Return signal of bearing1, bearing2, bearing3, bearing4
        return np.array(signal_df[0]), np.array(signal_df[1]), np.array(signal_df[2]), np.array(signal_df[3])

    def read_3rd_experiment(self):
        snap_shot_list = os.listdir(os.path.join(self.download_path, '4th_test', 'txt'))
        bearing3_1_list, bearing3_2_list, bearing3_3_list, bearing3_4_list = [], [], [], []
        for i, snap_shot in enumerate(snap_shot_list):
            # bearing3_1_snapshot, bearing3_2_snapshot, bearing3_3_snapshot, bearing3_4_snapshot = self.read_2nd_file(snap_shot)
            _, _, bearing3_3_snapshot, _ = self.read_3rd_file(snap_shot)

            # bearing3_1_list.append(bearing3_1_snapshot)
            # bearing3_2_list.append(bearing3_2_snapshot)
            bearing3_3_list.append(bearing3_3_snapshot)
            # bearing3_4_list.append(bearing3_4_snapshot)
            
        # Convert the list into a np array
        # bearing3_1 = np.vstack(bearing3_1_list)
        # bearing3_2 = np.vstack(bearing3_2_list)
        bearing3_3 = np.vstack(bearing3_3_list)
        # bearing3_4 = np.vstack(bearing3_4_list)
        
        # return bearing3_1, bearing3_2, bearing3_3, bearing3_4
        return None, None, bearing3_3, None

    def read_3rd_file(self, file_name):
        file_path = os.path.join(self.download_path, '4th_test', 'txt', file_name)
        signal_df = pd.read_csv(file_path, sep='\t', header=None)
        # Return signal of bearing1, bearing2, bearing3, bearing4
        return np.array(signal_df[0]), np.array(signal_df[1]), np.array(signal_df[2]), np.array(signal_df[3])