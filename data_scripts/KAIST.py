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