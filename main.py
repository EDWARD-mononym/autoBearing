import argparse

from data_scripts.FEMTO import FEMTO
from data_scripts.CWRU import CWRU
from data_scripts.HITSM import HITSM
from data_scripts.UNSW import UNSW
from data_scripts.FEMTO_fewshot import FEMTO_fewshot
from data_scripts.KAIST import KAIST
from data_scripts.MFPT import MFPT
from data_scripts.PU import PU

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def list_of_string(arg):
    return arg.split(',')

dataset_dict = {
    'CWRU': CWRU,
    'FEMTO': FEMTO,
    'FEMTO_fewshot': FEMTO_fewshot,
    'HITSM': HITSM,
    'UNSW': UNSW,
    'KAIST': KAIST,
    'MFPT': MFPT,
    'PU': PU
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Automatic bearing data downloader & processing')

    parser.add_argument('--dataset', default='PU', type=list_of_string, help='dataset to prepare')

    parser.add_argument('--raw_dir', default='raw_data', type=str, help='directory for downloaded raw data')
    parser.add_argument('--processed_dir', default='processed_data', type=str, help='directory for processed data')

    parser.add_argument('--window_size', default='1024', type=int)
    parser.add_argument('--stride', default='0.2', type=float)
    parser.add_argument('--few_shots', default='0.01,0.05', type=list_of_floats)
    parser.add_argument('--train_size', default='0.6', type=float)
    parser.add_argument('--val_size', default='0.2', type=float)
    parser.add_argument('--test_size', default='0.2', type=float)


    parser.add_argument('--fttp', default='True', type=bool)
    parser.add_argument('--normalise', default='True', type=bool)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    for dataset in args.dataset:
        class_object = dataset_dict[dataset]
        dataset_class = class_object(args)
        dataset_class.process_data()