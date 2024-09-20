from bs4 import BeautifulSoup
import os
from pathlib import Path
import requests
import scipy.io
import torch

from utils import download_file

class CWRU():
    def __init__(self, raw_dir, processed_dir) -> None:
        self.normal_baseline_link = 'https://engineering.case.edu/bearingdatacenter/normal-baseline-data'
        self.drive_end_12k_link = 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'
        self.download_path = Path(f'{raw_dir}/CWRU')
        self.processed_path = Path(f'{processed_dir}/CWRU')

    def process_data(self):
        self.download_data()

        if not os.path.exists(os.path.join(self.processed_path, 'train.pt')):
            healthy = ['Normal_0', 'Normal_1', 'Normal_2', 'Normal_3']
            faulty = ['IR007_0', 'IR007_1', 'IR007_2', 'IR007_3',
                      'B007_0', 'B007_1', 'B007_2', 'B007_3',
                      'OR007@6_0', 'OR007@6_1', 'OR007@6_2', 'OR007@6_3',
                      'IR014_0', 'IR014_1', 'IR014_2', 'IR014_3',
                      'B014_0', 'B014_1', 'B014_2', 'B014_3',
                      'OR014@6_0', 'OR014@6_1', 'OR014@6_2', 'OR014@6_3',
                      'IR021_0', 'IR021_1', 'IR021_2', 'IR021_3',
                      'B021_0', 'B021_1', 'B021_2', 'B021_3',
                      'OR021@6_0', 'OR021@6_1', 'OR021@6_2', 'OR021@6_3']

            for healthy_file in healthy:
                healthy_mat_name = self.mat_file_name[healthy_file]

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
                self.read_mat(os.path.join(self.download_path, file_name))
            
            except:
                print(f'Downloading {file_name}')
                download_url = mat_links[variable_name]
                download_file(download_url, self.download_path)

    def read_mat(self, file_name):
        return scipy.io.loadmat(file_name)

    # Functions to extract the data from the .mat file
    def read_cwru_mat_file(self, mat_file):
        mat_file_path = os.path.join(self.download_path, mat_file)
        mat_dict = self.read_mat(mat_file_path)
        de_key, fe_key = self.find_right_keys(mat_dict)
        de_data, fe_data = mat_dict[de_key], mat_dict[fe_key]
        de_tensor, fe_tensor = torch.tensor(de_data).unsqueeze(1), torch.tensor(fe_data).unsqueeze(1)
        
        return torch.concatenate((de_tensor, fe_tensor), axis=1)
        
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
