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

from utils import download_file, sliding_window_subsample, normalise_tensor, subsample_fewshots

class PU():
    def __init__(self, args) -> None:
        self.raw_zip_link = 'https://groups.uni-paderborn.de/kat/BearingDataCenter/'
        self.download_path = Path(f'{args.raw_dir}/PU')
        self.processed_path = Path(f'{args.processed_dir}/PU')

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
        if not os.path.exists(os.path.join(self.download_path, 'K001.rar')):
            #* Getting all .rar file links in self.raw_zip_link page 
            file_links = []
            response = requests.get(self.raw_zip_link)
            html = response.content
            soup = BeautifulSoup(html, "html.parser")
            for bs4Tag in soup.find_all('a'):
                link = bs4Tag.get('href')
                if link.endswith('.rar'):
                    download_link = self.raw_zip_link + link #? This should be form https://groups.uni-paderborn.de/kat/BearingDataCenter/K001.rar etc
                    file_links.append(download_link)

            print('Downloading PU raw files')
            for link in file_links:
                download_file(link, self.download_path)
                
        if not os.path.exists(os.path.join(self.download_path, 'K001')):
            for file in os.listdir(self.download_path):
                if file.endswith('.rar'):
                    patoolib.extract_archive(os.path.join(self.download_path, file), outdir=self.download_path)


    def process_data(self):
        self.download_data() # Download PU raw data if it doesn't exist in self.download_path

        if not os.path.exists(os.path.join(self.processed_path)):
            # healthy = ['K001', 'K002', 'K003', 'K004', 'K005', 'K006'] #! To mimic the data used in EverAdapt paper, only K001 is used for healthy samples
            healthy = ['K001']
            real_damages = ["KA04", "KB23", "KB27", "KI04"]
            artificial_damages = ["KA01", "KA03", "KA05", "KA07", "KI01", "KI03", "KI07"]

            self.create_dataset(healthy, real_damages, 'R')
            self.create_dataset(healthy, artificial_damages, 'A')

    def create_dataset(self, healthy, damages, dataset_type): # dataset_type is 'A' or 'R' and used for saving later on
        # Creating artificial dataset
        list_of_bearings_used = healthy + damages

        combined_1, combined_2, combined_3, combined_4 = [], [], [], []
        y_combined = []

        for label, bearing_folder in enumerate(list_of_bearings_used):
            # folder_path = os.path.join(self.download_path, bearing_folder)
            signal_1, signal_2, signal_3, signal_4 = self.load_mat_file_in_folder_separate_conditions(bearing_folder) # Signal should be in the shape torch.Size([20, 1, 256823])

            # creating labels, the objective is to clasify the severity of damage
            # thus common practice is to have signal from each bearing its own class
            y = torch.full([len(signal_1)], label) # Should have len(20)

            combined_1.append(signal_1)
            combined_2.append(signal_2)
            combined_3.append(signal_3)
            combined_4.append(signal_4)
            y_combined.append(y)
        
        combined_1 = torch.concatenate(combined_1, dim=0)
        combined_2 = torch.concatenate(combined_2, dim=0)
        combined_3 = torch.concatenate(combined_3, dim=0)
        combined_4 = torch.concatenate(combined_4, dim=0)
        y_combined = torch.concatenate(y_combined, dim=0)

        #* STEP 1
        # First split the signal into train and testing by slicing. This prevents information leekage from train to test
        signal_length = combined_1.shape[2] 
        train_size = int(signal_length * self.train_size)
        test_size = int(signal_length * self.test_size)
        val_size = signal_length - train_size - test_size

        combined_1_train = combined_1[:,:,:train_size]
        combined_1_val = combined_1[:,:,train_size:train_size+val_size]
        combined_1_test = combined_1[:,:,train_size+val_size:]

        combined_2_train = combined_2[:,:,:train_size]
        combined_2_val = combined_2[:,:,train_size:train_size+val_size]
        combined_2_test = combined_2[:,:,train_size+val_size:]

        combined_3_train = combined_3[:,:,:train_size]
        combined_3_val = combined_3[:,:,train_size:train_size+val_size]
        combined_3_test = combined_3[:,:,train_size+val_size:]

        combined_4_train = combined_4[:,:,:train_size]
        combined_4_val = combined_4[:,:,train_size:train_size+val_size]
        combined_4_test = combined_4[:,:,train_size+val_size:]

        #* STEP 2
        # Subsample with sliding window
        combined_1_train_signal, combined_1_train_label = sliding_window_subsample(combined_1_train, y_combined, self.window_size, self.step)
        combined_1_val_signal, combined_1_val_label = sliding_window_subsample(combined_1_val, y_combined, self.window_size, self.step)
        combined_1_test_signal, combined_1_test_label = sliding_window_subsample(combined_1_test, y_combined, self.window_size, self.step)

        combined_2_train_signal, combined_2_train_label = sliding_window_subsample(combined_2_train, y_combined, self.window_size, self.step)
        combined_2_val_signal, combined_2_val_label = sliding_window_subsample(combined_2_val, y_combined, self.window_size, self.step)
        combined_2_test_signal, combined_2_test_label = sliding_window_subsample(combined_2_test, y_combined, self.window_size, self.step)

        combined_3_train_signal, combined_3_train_label = sliding_window_subsample(combined_3_train, y_combined, self.window_size, self.step)
        combined_3_val_signal, combined_3_val_label = sliding_window_subsample(combined_3_val, y_combined, self.window_size, self.step)
        combined_3_test_signal, combined_3_test_label = sliding_window_subsample(combined_3_test, y_combined, self.window_size, self.step)

        combined_4_train_signal, combined_4_train_label = sliding_window_subsample(combined_4_train, y_combined, self.window_size, self.step)
        combined_4_val_signal, combined_4_val_label = sliding_window_subsample(combined_4_val, y_combined, self.window_size, self.step)
        combined_4_test_signal, combined_4_test_label = sliding_window_subsample(combined_4_test, y_combined, self.window_size, self.step)

        #* STEP 3
        # Save the dataset
        train_1 = {"samples": combined_1_train_signal, "labels": combined_1_train_label}
        val_1 = {"samples": combined_1_val_signal, "labels": combined_1_val_label}
        test_1 = {"samples": combined_1_test_signal, "labels": combined_1_test_label}

        train_2 = {"samples": combined_2_train_signal, "labels": combined_2_train_label}
        val_2 = {"samples": combined_2_val_signal, "labels": combined_2_val_label}
        test_2 = {"samples": combined_2_test_signal, "labels": combined_2_test_label}

        train_3 = {"samples": combined_3_train_signal, "labels": combined_3_train_label}
        val_3 = {"samples": combined_3_val_signal, "labels": combined_3_val_label}
        test_3 = {"samples": combined_3_test_signal, "labels": combined_3_test_label}

        train_4 = {"samples": combined_4_train_signal, "labels": combined_4_train_label}
        val_4 = {"samples": combined_4_val_signal, "labels": combined_4_val_label}
        test_4 = {"samples": combined_4_test_signal, "labels": combined_4_test_label}

        if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path)

        torch.save(train_1, os.path.join(self.processed_path, f"train_{dataset_type}1.pt"))
        torch.save(val_1, os.path.join(self.processed_path, f"val_{dataset_type}1.pt"))
        torch.save(test_1, os.path.join(self.processed_path, f"test_{dataset_type}1.pt"))

        torch.save(train_2, os.path.join(self.processed_path, f"train_{dataset_type}2.pt"))
        torch.save(val_2, os.path.join(self.processed_path, f"val_{dataset_type}2.pt"))
        torch.save(test_2, os.path.join(self.processed_path, f"test_{dataset_type}2.pt"))

        torch.save(train_3, os.path.join(self.processed_path, f"train_{dataset_type}3.pt"))
        torch.save(val_3, os.path.join(self.processed_path, f"val_{dataset_type}3.pt"))
        torch.save(test_3, os.path.join(self.processed_path, f"test_{dataset_type}3.pt"))

        torch.save(train_4, os.path.join(self.processed_path, f"train_{dataset_type}4.pt"))
        torch.save(val_4, os.path.join(self.processed_path, f"val_{dataset_type}4.pt"))
        torch.save(test_4, os.path.join(self.processed_path, f"test_{dataset_type}4.pt"))


    def load_mat_file_in_folder_combined_conditions(self):
        pass

    def load_mat_file_in_folder_separate_conditions(self, bearing_folder):

        mat_file_list = []
        folder = os.path.join(self.download_path, bearing_folder)
        for file in os.listdir(folder):
            if file.endswith('.mat'):
                mat_file_list.append(file)

        # Separating the file according to its working conditions
        # There are 20 files for each working condition and because the list is already sorted, we divide the list into 4 sections of 20 to take files from each condition
        condition_1 = mat_file_list[0:20]
        condition_2 = mat_file_list[20:40]
        condition_3 = mat_file_list[40:60]
        condition_4 = mat_file_list[60:80]

        signal_1 = self.read_list_of_mat_files(bearing_folder, condition_1)
        signal_2 = self.read_list_of_mat_files(bearing_folder, condition_2)
        signal_3 = self.read_list_of_mat_files(bearing_folder, condition_3)
        signal_4 = self.read_list_of_mat_files(bearing_folder, condition_4)

        return signal_1, signal_2, signal_3, signal_4

    def read_list_of_mat_files(self, folder, mat_file_list):
        combined_signal = []
        for mat_file in mat_file_list:
            signal = self.read_one_mat_file(folder, mat_file)
            combined_signal.append(signal)
        combined_signal = torch.stack(combined_signal, dim=0) # Should be in the shape torch.Size([20, 1, 249600])
        return combined_signal

    def read_one_mat_file(self, folder, mat_file_path):
        mat_file = loadmat(os.path.join(self.download_path, folder, mat_file_path))
        mat_file_name = str(mat_file_path)[:-4]

        vibration_data = mat_file[mat_file_name]["Y"][0][0][0][6][2] #? This is accessing the proper vibration signal in the matfile
        vibration_data = torch.tensor(vibration_data) # Should be in the shape torch.Size([1, L] where L varies)
        vibration_data = vibration_data[:,:249600] # Cuts the signal to 249600 datapoints. 249600 = 64k Sampling rate for 3.9 seconds

        return vibration_data



