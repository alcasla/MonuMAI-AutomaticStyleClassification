import argparse
import pickle
from random import randint
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

    # prepare the input data to be modeled
    train_labels = metadata[:, -1]      # select last column
    train_data = metadata[:, :metadata.shape[1]-1]      # select all except last column

    # convert the label to one-hot encode
    train_labels_int = train_labels.astype(int)     # convert class label to int
    num_classes = max(train_labels_int)+1       # max class index plus one (start from 0)
    hotone_train_labels = keras.utils.to_categorical(train_labels_int, num_classes)

    print( 'TRAIN data:', train_data.shape )
    num_sample = randint(0, train_labels.shape[0]-1)
    print( '\tsample ', train_data[num_sample, :], 'class:', train_labels_int[num_sample], hotone_train_labels[num_sample] )

    ###############################################

    # neural networks hyperparameters
    #param

    # NN training, validation/test code

    # https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3




######**********#######  MAIN  #########**********########
if __name__ == '__main__':
    main()