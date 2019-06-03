import argparse
import datetime
from tensorflow import keras
from tensorflow.python.keras.models import model_from_json

from tools.pickle_tools import *


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
    filenames, train_data, train_label = prepare_metadatada_from_pkl_file(train)
    # load test metadata structure from pickle file
    filenames, test_data, test_label = prepare_metadatada_from_pkl_file(test)


    ############## Multi Label Perceptron ##############
    # neural network hyperparameters
    num_classes = train_label.shape[1]      # hot-one encode - one column for each class
    num_epochs = 48
    batch_size = 64


    model = keras.Sequential()
    model.add(keras.layers.Dense(units=11, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(units=num_classes, activation='softmax'))
    #model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, train_label, batch_size=batch_size, epochs=num_epochs, verbose=0)
    # evaluate model and show metrics
    loss, accuracy = model.evaluate(test_data, test_label, verbose=1)
    print('Test loss: ', loss, '\tTest accuracy: ', accuracy)

    model_path = 'models/trainingJSON'
    timing = datetime.datetime.now().strftime('_%Y-%m-%d_%H:%M:%S')
    # serialize model to json and save
    model_json = model.to_json()
    with open(model_path + timing + '.json', 'w') as json:
        json.write(model_json)
    # serialize weights to HDF5 and save
    model.save_weights(model_path + timing + '.h5')
    print('Model saved in:', model_path+timing)


######**********#######  MAIN  #########**********########
if __name__ == '__main__':
    main()