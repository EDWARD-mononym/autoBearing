import argparse

from data_scripts.FEMTO import FEMTO
from data_scripts.IMS import IMS


def parse_arguments():
    parser = argparse.ArgumentParser(description='Automatic bearing data downloader & processing')

    parser.add_argument('--raw_dir', default='raw_data', type=str, help='directory for downloaded raw data')
    parser.add_argument('--processed_dir', default='processed_data', type=str, help='directory for processed data')

    parser.add_argument('--window_size', default='1024', type=int)
    parser.add_argument('--stride', default='0.2', type=float)

    parser.add_argument('--fttp', default='True', type=bool)
    parser.add_argument('--normalise', default='True', type=bool)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    femto_class = FEMTO(args)
    femto_class.process_data()