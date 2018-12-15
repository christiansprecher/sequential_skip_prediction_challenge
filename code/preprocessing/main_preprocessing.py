import os
import argparse
from datagen import Datagen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)

    io_args = parser.parse_args()
    folder = io_args.folder

    data = Datagen(folder)

    data.load_training_data()
