from bs4 import BeautifulSoup
import numpy as np
import os
from pathlib import Path
import requests
import scipy.io
import torch

from utils import download_file, set_seed_and_deterministic, train_test_split, sliding_window_subsample, subsample_fewshots

class CWRU_FD():
    def __init__(self, args) -> None:
        self.normal_baseline_link = 'https://engineering.case.edu/bearingdatacenter/normal-baseline-data'
        self.drive_end_12k_link = 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'
        self.download_path = Path(f'{args.raw_dir}/CWRU')
        self.processed_path = Path(f'{args.processed_dir}/CWRU_FD')

        self.window_size = args.window_size
        self.stride = args.stride
        self.step = int(self.window_size * self.stride)
        self.few_shots = args.few_shots

        self.train_size = args.train_size
        self.test_size = args.test_size
        self.val_size = args.val_size

    def process_data(self):
        self.download_data()

        if not os.path.exists(os.path.join(self.processed_path, 'train.pt')):
            healthy = ['Normal_0', 'Normal_1', 'Normal_2', 'Normal_3']
            inner_damage = ['IR007_0', 'IR007_1', 'IR007_2', 'IR007_3', 'IR014_0', 'IR014_1', 'IR014_2', 'IR014_3', 'IR021_0', 'IR021_1', 'IR021_2', 'IR021_3']
            outer_damage = ['OR007@6_0', 'OR007@6_1', 'OR007@6_2', 'OR007@6_3', 'OR014@6_0', 'OR014@6_1', 'OR014@6_2', 'OR014@6_3', 'OR021@6_0', 'OR021@6_1', 'OR021@6_2', 'OR021@6_3']

            healthy_signals, inner_signals, outer_signals = self.read_list_of_bearings(healthy), self.read_list_of_bearings(inner_damage), self.read_list_of_bearings(outer_damage)

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

            # # Generate fewshots
            # for few_shot_size in self.few_shots:
            #     x_few, y_few = subsample_fewshots(train_x, train_y, few_shot_size)
            #     train_few_shot = {"samples": x_few, "labels": y_few}
            #     if not os.path.exists(self.processed_path):
            #         os.makedirs(self.processed_path)
            #     torch.save(train_few_shot, os.path.join(self.processed_path, f"train_few_shot_{str(few_shot_size).split('.')[1]}.pt"))

            # Save data
            train = {"samples": train_x, "labels": train_y}
            val = {"samples": val_x, "labels": val_y}
            test = {"samples": test_x, "labels": test_y}
            if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path)
            torch.save(train, os.path.join(self.processed_path, "train.pt"))
            torch.save(val, os.path.join(self.processed_path, "val.pt"))
            torch.save(test, os.path.join(self.processed_path, "test.pt"))

    def download_data(self):
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        #* ############################################################################################
        #* Get all mat lab links from normal_baseline & drive_end_12k
        #* ############################################################################################
        self.mat_file_name, mat_links = {}, {}

        response = requests.get(self.normal_baseline_link)
        html = response.content
        soup = BeautifulSoup(html, "html.parser")
        for bs4Tag in soup.find_all('a'):
            link = bs4Tag.get('href')
            if link:
                if link.endswith('.mat'):
                    variable_name = bs4Tag.text.strip() #? variable name denotes Normal_1, Normal_2, etc
                    file_name = Path(link).name #? file_name should be in the form 97, 102, etc
                    mat_links[variable_name] = link
                    self.mat_file_name[variable_name] = file_name

        response = requests.get(self.drive_end_12k_link)
        html = response.content
        soup = BeautifulSoup(html, "html.parser")
        for bs4Tag in soup.find_all('a'):
            link = bs4Tag.get('href')
            if link:
                if link.endswith('.mat'):
                    variable_name = bs4Tag.text.strip() #? variable name denotes Normal_1, Normal_2, etc
                    file_name = Path(link).name #? file_name should be in the form 97, 102, etc
                    mat_links[variable_name] = link
                    self.mat_file_name[variable_name] = file_name

        #* ############################################################################################
        #* Download all mat lab files if it hasn't already been downloaded yet
        #* ############################################################################################
        print('Downloading CWRU raw files')
        for variable_name, file_name in self.mat_file_name.items():
            try:
                self.read_mat(file_name)
            
            except:
                file_number = int(file_name.split('.')[0])
                if file_number >= 3000:
                    continue
                print(f'Downloading {file_name}')
                download_url = mat_links[variable_name]
                download_file(download_url, self.download_path)

    def read_list_of_bearings(self, list_of_bearings):
        x_list, y_list = [], []
        for bearing_name in list_of_bearings:
            bearing_mat_name = self.mat_file_name[bearing_name]
            x, y = self.read_mat(bearing_mat_name)
            x_list.append(x)
            y_list.append(y)

        # Convert the list into a np array
        x_signal, y_signal = np.vstack(x_list), np.vstack(y_list)
        x_tensor, y_tensor = torch.tensor(x_signal).unsqueeze(1), torch.tensor(y_signal).unsqueeze(1)

        return torch.concatenate((x_tensor, y_tensor), axis=1)

    # Functions to extract the data from the .mat file
    def read_mat(self, mat_file):
        mat_file_path = os.path.join(self.download_path, mat_file)
        mat_dict = scipy.io.loadmat(mat_file_path)
        de_key, fe_key = self.find_right_keys(mat_dict)
        de_data, fe_data = mat_dict[de_key], mat_dict[fe_key]
        de_data, fe_data = de_data[:120000], fe_data[:120000] # 12k sampling rate for 10 seconds

        return de_data.T, fe_data.T
        
    def find_right_keys(self, mat_dict): #? Finds the right key to get the DE and FE data
        de_key, fe_key = None, None
        for key in mat_dict:
            if key[-7:] == 'DE_time':
                de_key = key
            elif key[-7:] == 'FE_time':
                fe_key = key
            else:
                pass
        return de_key, fe_key
