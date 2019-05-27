import argparse
import pickle
from tensorflow import keras


def main():
    ####### Parameter parsing #######
    parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
    parser.add_argument('--pkl', dest='pkl', help='File path to save pickle file.', required=True)
    args = parser.parse_args()
    #################################

    input_file = args.pkl

    ############## Data preparation ##############
    # load metadata structure from pickle file
    with(open(input_file, 'rb')) as pkl:
        metadata = pickle.load(pkl)
        print('Pickle file readed: ', input_file, metadata.shape)

    # prepare the aggregation element vector to be modeled
    # prepare image categorical label - one-hot encoding

    # https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3

    # neural networks hyperparameters
    #param

    # NN training, validation/test code




######**********#######  MAIN  #########**********########
if __name__ == '__main__':
    main()