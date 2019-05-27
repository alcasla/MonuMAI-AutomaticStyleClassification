import argparse
import pickle

from tools.metadata_tools import *


def main():
    ####### Parameter parsing #######
    parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
    parser.add_argument('--pkl', dest='pkl', help='File path to save pickle file.', required=True)
    args = parser.parse_args()
    #################################

    input_file = args.pkl

    # load metadata structure from pickle file
    with(open(input_file, 'rb')) as pkl:
        metadata = pickle.load(pkl)
        print('Pickle file readed: ', type(metadata), metadata.shape)

    # neural networks hyperparameters
    #param

    # NN training, validation/test code



######**********#######  MAIN  #########**********########
if __name__ == '__main__':
    main()