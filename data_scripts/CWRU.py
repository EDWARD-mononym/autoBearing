from bs4 import BeautifulSoup
import os
from pathlib import Path
import requests
import scipy.io
from tqdm.auto import tqdm
from utils import set_seed_and_deterministic, download_file

class CWRU():
    def __init__(self, raw_dir, processed_dir) -> None:
        self.normal_baseline_link = 'https://engineering.case.edu/bearingdatacenter/normal-baseline-data'
        self.drive_end_12k_link = 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'
        self.download_path = Path(f'{raw_dir}/CWRU')
        self.processed_path = Path(f'{processed_dir}/CWRU')

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
