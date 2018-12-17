import os
import argparse
from datagen import Datagen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)
    parser.add_argument('-v', '--verbosity', required=False, default=1)

    io_args = parser.parse_args()
    folder = io_args.folder
    verbosity = io_args.verbosity


    data = Datagen(folder,verbosity)

    data.load_training_data()
