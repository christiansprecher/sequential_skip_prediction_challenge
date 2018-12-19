import os
import argparse
from datagen import Datagen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)
    parser.add_argument('-v', '--verbosity', required=False, default=1)
    parser.add_argument('-r', '--training', required=False, default=0)
    parser.add_argument('-t', '--test', required=False, default=0)
    parser.add_argument('-o', '--overwrite', required=False, default=1)

    io_args = parser.parse_args()
    folder = io_args.folder
    verbosity = io_args.verbosity
    training = io_args.training
    test = io_args.test
    overwrite = io_args.overwrite

    if test or training:
        data = Datagen(folder,overwrite,verbosity)
        if(training):
            data.load_training_data()
        if(test):
            data.load_test_data()
    else:
        parser.print_help()
