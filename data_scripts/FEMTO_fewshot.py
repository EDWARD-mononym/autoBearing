import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
import shutil
import zipfile

from utils import download_file, sliding_window_subsample, normalise_tensor, subsample_fewshots

class FEMTO_fewshot():
    def __init__(self, args) -> None:
        self.raw_zip_link = 'https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset/archive/refs/heads/master.zip'
        self.download_path = Path(f'{args.raw_dir}/FEMTO')
        self.processed_path = Path(f'{args.processed_dir}/FEMTO_fewshot')

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
        if not os.path.exists(os.path.join(self.download_path, 'master.zip')):
            print('Downloading FEMTO raw files')
            download_file(self.raw_zip_link, self.download_path)
            
        if not os.path.exists(os.path.join(self.download_path, 'phm-ieee-2012-data-challenge-dataset-master')):  
            with zipfile.ZipFile(os.path.join(self.download_path, 'master.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.download_path)

        if not os.path.exists(os.path.join(self.download_path, 'Full_Test_Set')):
            for file_name in os.listdir(os.path.join(self.download_path, 'phm-ieee-2012-data-challenge-dataset-master')):
                shutil.move(os.path.join(os.path.join(self.download_path, 'phm-ieee-2012-data-challenge-dataset-master'), file_name), self.download_path)

        if not os.path.exists(os.path.join(self.download_path, 'Bearing1_1')):
            for file_name in os.listdir(os.path.join(self.download_path, 'Learning_set')):
                shutil.move(os.path.join(os.path.join(self.download_path, 'Learning_set'), file_name), self.download_path)
            for file_name in os.listdir(os.path.join(self.download_path, 'Full_Test_Set')):
                shutil.move(os.path.join(os.path.join(self.download_path, 'Full_Test_Set'), file_name), self.download_path)

    def process_data(self):
        self.download_data()

        if not os.path.exists(os.path.join(self.processed_path, 'test.pt')):
            bearing1_1 = self.read_signal('bearing1_1')
            bearing1_2 = self.read_signal('bearing1_2')
            bearing1_3 = self.read_signal('bearing1_3')
            bearing1_4 = self.read_signal('bearing1_4')
            bearing1_5 = self.read_signal('bearing1_5')
            bearing1_6 = self.read_signal('bearing1_6')
            bearing1_7 = self.read_signal('bearing1_7')

            if self.fttp: #* If true, cut the signal so that only the wear-out period is going to be predicted
                bearing1_1 = bearing1_1[407:]
                bearing1_2 = bearing1_2[544:]
                bearing1_3 = bearing1_3[521:]
                bearing1_4 = bearing1_4[840:]
                bearing1_5 = bearing1_5[2306:]
                bearing1_6 = bearing1_6[479:]
                bearing1_7 = bearing1_7[995:]

            if self.normalise: #* If true, normalise the signal so that it has mean=0 and s.d=1
                bearing1_1 = normalise_tensor(bearing1_1)
                bearing1_2 = normalise_tensor(bearing1_2)
                bearing1_3 = normalise_tensor(bearing1_3)
                bearing1_4 = normalise_tensor(bearing1_4)
                bearing1_5 = normalise_tensor(bearing1_5)
                bearing1_6 = normalise_tensor(bearing1_6)
                bearing1_7 = normalise_tensor(bearing1_7)


            #* Create RUL by counting how many timestep a snapshot has until the last snapshot
            bearing1_1_y = torch.tensor(np.arange(len(bearing1_1)-1, -1, -1))
            bearing1_2_y = torch.tensor(np.arange(len(bearing1_2)-1, -1, -1))
            bearing1_3_y = torch.tensor(np.arange(len(bearing1_3)-1, -1, -1))
            bearing1_4_y = torch.tensor(np.arange(len(bearing1_4)-1, -1, -1))
            bearing1_5_y = torch.tensor(np.arange(len(bearing1_5)-1, -1, -1))
            bearing1_6_y = torch.tensor(np.arange(len(bearing1_6)-1, -1, -1))
            bearing1_7_y = torch.tensor(np.arange(len(bearing1_7)-1, -1, -1))

            #* Convert RUL from timestep unit to % health remaining
            bearing1_1_y = bearing1_1_y / (len(bearing1_1_y)-1)
            bearing1_2_y = bearing1_2_y / (len(bearing1_2_y)-1)
            bearing1_3_y = bearing1_3_y / (len(bearing1_3_y)-1)
            bearing1_4_y = bearing1_4_y / (len(bearing1_4_y)-1)
            bearing1_5_y = bearing1_5_y / (len(bearing1_5_y)-1)
            bearing1_6_y = bearing1_6_y / (len(bearing1_6_y)-1)
            bearing1_7_y = bearing1_7_y / (len(bearing1_7_y)-1)

            # Subsample the signal such that it has a length of self.window_size
            bearing1_1, bearing1_1_y = sliding_window_subsample(bearing1_1, bearing1_1_y, self.window_size, self.step)
            bearing1_2, bearing1_2_y = sliding_window_subsample(bearing1_2, bearing1_2_y, self.window_size, self.step)
            bearing1_3, bearing1_3_y = sliding_window_subsample(bearing1_3, bearing1_3_y, self.window_size, self.step)
            bearing1_4, bearing1_4_y = sliding_window_subsample(bearing1_4, bearing1_4_y, self.window_size, self.step)
            bearing1_5, bearing1_5_y = sliding_window_subsample(bearing1_5, bearing1_5_y, self.window_size, self.step)
            bearing1_6, bearing1_6_y = sliding_window_subsample(bearing1_6, bearing1_6_y, self.window_size, self.step)
            bearing1_7, bearing1_7_y = sliding_window_subsample(bearing1_7, bearing1_7_y, self.window_size, self.step)

            #* Generate few shot samples
            all_bearing_x = [bearing1_1, bearing1_2, bearing1_3, bearing1_4, bearing1_5, bearing1_6]
            all_bearing_y = [bearing1_1_y, bearing1_2_y, bearing1_3_y, bearing1_4_y, bearing1_5_y, bearing1_6_y]
            for i in range(6):
                if i:
                    bearing_list_x, bearing_list_y = all_bearing_x[:i+1], all_bearing_y[:i+1]
                    train_x, train_y = torch.cat(bearing_list_x, dim=0), torch.cat(bearing_list_y, dim=0)
                else:
                    train_x, train_y = bearing1_1, bearing1_1_y

                if not os.path.exists(self.processed_path):
                    os.makedirs(self.processed_path)
                train = {"samples": train_x, "labels": train_y}
                torch.save(train, os.path.join(self.processed_path, f"train_{i+1}.pt"))
    
            test_x, test_y = bearing1_7, bearing1_7_y
            test = {"samples": test_x, "labels": test_y}
            torch.save(test, os.path.join(self.processed_path, "test.pt"))

    def read_signal(self, bearing_name):
        x_list, y_list = [], []
        for file_name in os.listdir(os.path.join(self.download_path, bearing_name)):
            if file_name[:3] == 'acc':
                x, y = self.get_x_y(bearing_name, file_name)
                x_list.append(x)
                y_list.append(y)
            
        # Convert the list into a np array
        x_signal, y_signal = np.vstack(x_list), np.vstack(y_list)
        x_tensor, y_tensor = torch.tensor(x_signal).unsqueeze(1), torch.tensor(y_signal).unsqueeze(1)
        
        return torch.concatenate((x_tensor, y_tensor), axis=1)

    def get_x_y(self, bearing_name, file_name):
        signal = pd.read_csv(os.path.join(self.download_path,  bearing_name, file_name), header=None)
        if len(signal.columns) == 1:
            signal = pd.read_csv(os.path.join(self.download_path,  bearing_name, file_name), header=None, delimiter=';')
        return np.array(signal[4]), np.array(signal[5])