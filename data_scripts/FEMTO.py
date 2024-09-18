import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
import shutil
import zipfile

from utils import download_file, sliding_window_subsample, normalise_tensor, subsample_fewshots

class FEMTO():
    def __init__(self, args) -> None:
        self.raw_zip_link = 'https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset/archive/refs/heads/master.zip'
        self.download_path = Path(f'{args.raw_dir}/FEMTO')
        self.processed_path = Path(f'{args.processed_dir}/FEMTO')

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

        if not os.path.exists(os.path.join(self.processed_path, 'train.pt')):
            bearing1_1 = self.read_signal('bearing1_1')
            bearing1_2 = self.read_signal('bearing1_2')
            bearing1_3 = self.read_signal('bearing1_3')
            bearing1_4 = self.read_signal('bearing1_4')
            bearing1_5 = self.read_signal('bearing1_5')
            bearing1_6 = self.read_signal('bearing1_6')
            bearing1_7 = self.read_signal('bearing1_7')

            bearing2_1 = self.read_signal('bearing2_1')
            bearing2_2 = self.read_signal('bearing2_2')
            bearing2_3 = self.read_signal('bearing2_3')
            bearing2_4 = self.read_signal('bearing2_4')
            bearing2_5 = self.read_signal('bearing2_5')
            bearing2_6 = self.read_signal('bearing2_6')
            bearing2_7 = self.read_signal('bearing2_7')

            bearing3_1 = self.read_signal('bearing3_1')
            bearing3_2 = self.read_signal('bearing3_2')
            bearing3_3 = self.read_signal('bearing3_3')

            if self.fttp: #* If true, cut the signal so that only the wear-out period is going to be predicted
                bearing1_1 = bearing1_1[407:]
                bearing1_2 = bearing1_2[544:]
                bearing1_3 = bearing1_3[521:]
                bearing1_4 = bearing1_4[840:]
                bearing1_5 = bearing1_5[2306:]
                bearing1_6 = bearing1_6[479:]
                bearing1_7 = bearing1_7[995:]

                bearing2_1 = bearing2_1[819:]
                bearing2_2 = bearing2_2[192:]
                bearing2_3 = bearing2_3[257:]
                bearing2_4 = bearing2_4[248:]
                bearing2_5 = bearing2_5[252:]
                bearing2_6 = bearing2_6[213:]
                bearing2_7 = bearing2_7[163:]

                bearing3_1 = bearing3_1[132:]
                bearing3_2 = bearing3_2[116:]
                bearing3_3 = bearing3_3[306:]

            if self.normalise: #* If true, normalise the signal so that it has mean=0 and s.d=1
                bearing1_1 = normalise_tensor(bearing1_1)
                bearing1_2 = normalise_tensor(bearing1_2)
                bearing1_3 = normalise_tensor(bearing1_3)
                bearing1_4 = normalise_tensor(bearing1_4)
                bearing1_5 = normalise_tensor(bearing1_5)
                bearing1_6 = normalise_tensor(bearing1_6)
                bearing1_7 = normalise_tensor(bearing1_7)

                bearing2_1 = normalise_tensor(bearing2_1)
                bearing2_2 = normalise_tensor(bearing2_2)
                bearing2_3 = normalise_tensor(bearing2_3)
                bearing2_4 = normalise_tensor(bearing2_4)
                bearing2_5 = normalise_tensor(bearing2_5)
                bearing2_6 = normalise_tensor(bearing2_6)
                bearing2_7 = normalise_tensor(bearing2_7)

                bearing3_1 = normalise_tensor(bearing3_1)
                bearing3_2 = normalise_tensor(bearing3_2)
                bearing3_3 = normalise_tensor(bearing3_3)

            #* Create RUL by counting how many timestep a snapshot has until the last snapshot
            bearing1_1_y = torch.tensor(np.arange(len(bearing1_1)-1, -1, -1))
            bearing1_2_y = torch.tensor(np.arange(len(bearing1_2)-1, -1, -1))
            bearing1_3_y = torch.tensor(np.arange(len(bearing1_3)-1, -1, -1))
            bearing1_4_y = torch.tensor(np.arange(len(bearing1_4)-1, -1, -1))
            bearing1_5_y = torch.tensor(np.arange(len(bearing1_5)-1, -1, -1))
            bearing1_6_y = torch.tensor(np.arange(len(bearing1_6)-1, -1, -1))
            bearing1_7_y = torch.tensor(np.arange(len(bearing1_7)-1, -1, -1))

            bearing2_1_y = torch.tensor(np.arange(len(bearing2_1)-1, -1, -1))
            bearing2_2_y = torch.tensor(np.arange(len(bearing2_2)-1, -1, -1))
            bearing2_3_y = torch.tensor(np.arange(len(bearing2_3)-1, -1, -1))
            bearing2_4_y = torch.tensor(np.arange(len(bearing2_4)-1, -1, -1))
            bearing2_5_y = torch.tensor(np.arange(len(bearing2_5)-1, -1, -1))
            bearing2_6_y = torch.tensor(np.arange(len(bearing2_6)-1, -1, -1))
            bearing2_7_y = torch.tensor(np.arange(len(bearing2_7)-1, -1, -1))

            bearing3_1_y = torch.tensor(np.arange(len(bearing3_1)-1, -1, -1))
            bearing3_2_y = torch.tensor(np.arange(len(bearing3_2)-1, -1, -1))
            bearing3_3_y = torch.tensor(np.arange(len(bearing3_3)-1, -1, -1))

            #* Convert RUL from timestep unit to % health remaining
            bearing1_1_y = bearing1_1_y / (len(bearing1_1_y)-1)
            bearing1_2_y = bearing1_2_y / (len(bearing1_2_y)-1)
            bearing1_3_y = bearing1_3_y / (len(bearing1_3_y)-1)
            bearing1_4_y = bearing1_4_y / (len(bearing1_4_y)-1)
            bearing1_5_y = bearing1_5_y / (len(bearing1_5_y)-1)
            bearing1_6_y = bearing1_6_y / (len(bearing1_6_y)-1)
            bearing1_7_y = bearing1_7_y / (len(bearing1_7_y)-1)

            bearing2_1_y = bearing2_1_y / (len(bearing2_1_y)-1)
            bearing2_2_y = bearing2_2_y / (len(bearing2_2_y)-1)
            bearing2_3_y = bearing2_3_y / (len(bearing2_3_y)-1)
            bearing2_4_y = bearing2_4_y / (len(bearing2_4_y)-1)
            bearing2_5_y = bearing2_5_y / (len(bearing2_5_y)-1)
            bearing2_6_y = bearing2_6_y / (len(bearing2_6_y)-1)
            bearing2_7_y = bearing2_7_y / (len(bearing2_7_y)-1)

            bearing3_1_y = bearing3_1_y / (len(bearing3_1_y)-1)
            bearing3_2_y = bearing3_2_y / (len(bearing3_2_y)-1)
            bearing3_3_y = bearing3_3_y / (len(bearing3_3_y)-1)


            # Subsample the signal such that it has a length of self.window_size
            bearing1_1, bearing1_1_y = sliding_window_subsample(bearing1_1, bearing1_1_y, self.window_size, self.step)
            bearing1_2, bearing1_2_y = sliding_window_subsample(bearing1_2, bearing1_2_y, self.window_size, self.step)
            bearing1_3, bearing1_3_y = sliding_window_subsample(bearing1_3, bearing1_3_y, self.window_size, self.step)
            bearing1_4, bearing1_4_y = sliding_window_subsample(bearing1_4, bearing1_4_y, self.window_size, self.step)
            bearing1_5, bearing1_5_y = sliding_window_subsample(bearing1_5, bearing1_5_y, self.window_size, self.step)
            bearing1_6, bearing1_6_y = sliding_window_subsample(bearing1_6, bearing1_6_y, self.window_size, self.step)
            bearing1_7, bearing1_7_y = sliding_window_subsample(bearing1_7, bearing1_7_y, self.window_size, self.step)

            bearing2_1, bearing2_1_y = sliding_window_subsample(bearing2_1, bearing2_1_y, self.window_size, self.step)
            bearing2_2, bearing2_2_y = sliding_window_subsample(bearing2_2, bearing2_2_y, self.window_size, self.step)
            bearing2_3, bearing2_3_y = sliding_window_subsample(bearing2_3, bearing2_3_y, self.window_size, self.step)
            bearing2_4, bearing2_4_y = sliding_window_subsample(bearing2_4, bearing2_4_y, self.window_size, self.step)
            bearing2_5, bearing2_5_y = sliding_window_subsample(bearing2_5, bearing2_5_y, self.window_size, self.step)
            bearing2_6, bearing2_6_y = sliding_window_subsample(bearing2_6, bearing2_6_y, self.window_size, self.step)
            bearing2_7, bearing2_7_y = sliding_window_subsample(bearing2_7, bearing2_7_y, self.window_size, self.step)

            bearing3_1, bearing3_1_y = sliding_window_subsample(bearing3_1, bearing3_1_y, self.window_size, self.step)
            bearing3_2, bearing3_2_y = sliding_window_subsample(bearing3_2, bearing3_2_y, self.window_size, self.step)
            bearing3_3, bearing3_3_y = sliding_window_subsample(bearing3_3, bearing3_3_y, self.window_size, self.step)

            for few_shot_size in self.few_shots:
                bearing1_1_few, bearing1_1_y_few = subsample_fewshots(bearing1_1, bearing1_1_y, few_shot_size)
                bearing1_2_few, bearing1_2_y_few = subsample_fewshots(bearing1_2, bearing1_2_y, few_shot_size)
                bearing2_1_few, bearing2_1_y_few = subsample_fewshots(bearing2_1, bearing2_1_y, few_shot_size)
                bearing2_2_few, bearing2_2_y_few = subsample_fewshots(bearing2_2, bearing2_2_y, few_shot_size)
                bearing3_1_few, bearing3_1_y_few = subsample_fewshots(bearing3_1, bearing3_1_y, few_shot_size)

                train_few_shot_x, train_few_shot_y = torch.cat((bearing1_1_few, bearing1_2_few, bearing2_1_few, bearing2_2_few, bearing3_1_few), dim=0), torch.cat((bearing1_1_y_few, bearing1_2_y_few, bearing2_1_y_few, bearing2_2_y_few, bearing3_1_y_few), dim=0)
                train_few_shot = {"samples": train_few_shot_x, "labels": train_few_shot_y}

                if not os.path.exists(self.processed_path):
                    os.makedirs(self.processed_path)
                torch.save(train_few_shot, os.path.join(self.processed_path, f"train_few_shot_{str(few_shot_size).split('.')[1]}.pt"))

            train_x, train_y = torch.cat((bearing1_1, bearing1_2, bearing2_1, bearing2_2, bearing3_1), dim=0), torch.cat((bearing1_1_y, bearing1_2_y, bearing2_1_y, bearing2_2_y, bearing3_1_y), dim=0)
            val_x, val_y = torch.cat((bearing1_3, bearing2_3, bearing3_2), dim=0), torch.cat((bearing1_3_y, bearing2_3_y, bearing3_2_y), dim=0)
            test_x, test_y = torch.cat((bearing1_4, bearing1_5, bearing1_6, bearing1_7, bearing2_4, bearing2_5, bearing2_6, bearing2_7, bearing3_3), dim=0), torch.cat((bearing1_4_y, bearing1_5_y, bearing1_6_y, bearing1_7_y, bearing2_4_y, bearing2_5_y, bearing2_6_y, bearing2_7_y, bearing3_3_y), dim=0)

            train = {"samples": train_x, "labels": train_y}
            val = {"samples": val_x, "labels": val_y}
            test = {"samples": test_x, "labels": test_y}

            if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path)
            torch.save(train, os.path.join(self.processed_path, "train.pt"))
            torch.save(val, os.path.join(self.processed_path, "val.pt"))
            torch.save(test, os.path.join(self.processed_path, "test.pt"))

    def read_signal(self, bearing_name):
        x_list, y_list = [], []
        for file_name in os.listdir(os.path.join(self.download_path, bearing_name)):
            if file_name[:3] == 'acc':
                x, y = self.get_x_y(bearing_name, file_name)
                x_list.append(x)
                y_list.append(x)
            
        # Convert the list into a np array
        x_signal, y_signal = np.vstack(x_list), np.vstack(y_list)
        x_tensor, y_tensor = torch.tensor(x_signal).unsqueeze(1), torch.tensor(y_signal).unsqueeze(1)
        
        return torch.concatenate((x_tensor, y_tensor), axis=1)

    def get_x_y(self, bearing_name, file_name):
        signal = pd.read_csv(os.path.join(self.download_path,  bearing_name, file_name), header=None)
        if len(signal.columns) == 1:
            signal = pd.read_csv(os.path.join(self.download_path,  bearing_name, file_name), header=None, delimiter=';')
        return np.array(signal[4]), np.array(signal[5])