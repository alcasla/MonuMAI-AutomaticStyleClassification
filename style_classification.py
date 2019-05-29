import argparse
import pickle
from random import randint
from tensorflow import keras


def read_metadatada_pkl_file(filepath):
    print('POR HACER')

def main():
    ####### Parameter parsing #######
    parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
    parser.add_argument('--trn', dest='train', help='File path to save pickle file.', required=True)
    parser.add_argument('--tst', dest='test', help='File path to save pickle file.', required=True)
    args = parser.parse_args()
    #################################

    train = args.train
    test = args.test

    ############## Data preparation ##############
    # load training metadata structure from pickle file
    with(open(train, 'rb')) as pkl:
        metadata = pickle.load(pkl)
        print('Pickle file readed: ', train, metadata.shape)

    # prepare the input data to be modeled
    train_labels = metadata[:, -1]      # select last column
    train_data = metadata[:, :metadata.shape[1]-1]      # select all except last column

    # convert the label to one-hot encode
    train_labels = train_labels.astype(int)     # convert class label to int
    num_classes = max(train_labels)+1       # max class index plus one (start from 0)
    hotone_train_labels = keras.utils.to_categorical(train_labels, num_classes)

    print( 'TRAIN data:', train_data.shape )
    num_sample = randint(0, train_labels.shape[0]-1)
    print( '\tsample ', train_data[num_sample, :], 'class:', train_labels[num_sample], hotone_train_labels[num_sample] )
    ##### Training: train_data - hotone_train_labels

    # load test metadata structure from pickle file
    with(open(test, 'rb')) as pkl:
        metadata = pickle.load(pkl)
        print('Pickle file readed: ', test, metadata.shape)

    # prepare the input data to be modeled
    test_labels = metadata[:, -1]  # select last column
    test_data = metadata[:, :metadata.shape[1] - 1]  # select all except last column

    # convert the label to one-hot encode
    test_labels = test_labels.astype(int)  # convert class label to int
    num_classes = max(test_labels) + 1  # max class index plus one (start from 0)
    hotone_test_labels = keras.utils.to_categorical(test_labels, num_classes)

    print('TEST data:', test_data.shape)
    num_sample = randint(0, test_labels.shape[0] - 1)
    print('\tsample ', test_data[num_sample, :], 'class:', test_labels[num_sample], hotone_test_labels[num_sample])
    ##### Test: test_data - hotone_test_labels

    ###############################################

    # neural networks hyperparameters
    num_epochs = 30
    batch_size = 64

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=11, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(units=num_classes, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, hotone_train_labels, batch_size=batch_size, epochs=num_epochs, verbose=1)
    loss, accuracy = model.evaluate(test_data, hotone_test_labels, verbose=1)
    print('Test loss: ', loss)
    print('Test accuracy: ', accuracy)

    # NN training, validation/test code

    # https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3




######**********#######  MAIN  #########**********########
if __name__ == '__main__':
    main()