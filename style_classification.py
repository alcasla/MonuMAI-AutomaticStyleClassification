import argparse
import pickle
from random import randint
from tensorflow import keras
from tensorflow.python.keras.models import model_from_json


def prepare_metadatada_from_pkl_file(filepath):
    with(open(filepath, 'rb')) as pkl:
        metadata = pickle.load(pkl)
        print('Pickle file readed: ', filepath, metadata.shape)

    # prepare the input data to be modeled
    labels = metadata[:, -1]  # select last column
    data = metadata[:, :metadata.shape[1] - 1]  # select all except last column

    # convert the label to one-hot encode
    labels = labels.astype(int)  # convert class label to int
    num_classes = max(labels) + 1  # max class index plus one (start from 0)
    hotone_labels = keras.utils.to_categorical(labels, num_classes)

    print('TRAIN data:', data.shape)
    num_sample = randint(0, labels.shape[0] - 1)
    print('\tsample ', data[num_sample, :], 'class:', labels[num_sample], hotone_labels[num_sample])

    return data, hotone_labels


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
    train_data, train_label = prepare_metadatada_from_pkl_file(train)
    # load test metadata structure from pickle file
    test_data, test_label = prepare_metadatada_from_pkl_file(test)


    ############## Multi Label Perceptron ##############
    # neural network hyperparameters
    num_classes = train_label.shape[1]      # hot-one encode - one column for each class
    num_epochs = 54
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

    # calculate predictions
    predictions = model.predict(test_data, batch_size=1)
    print( predictions )


    # serialize model to json and save
    model_json = model.to_json()
    with open('model.json', 'w') as json:
        json.write(model_json)
    # serialize weights to HDF5 and save
    model.save_weights('model.h5')
    print('Model saved')

    """
    #load model from json and create
    json = open('model.json', 'r')
    model_json = json.read()
    json.close()
    loaded_model = model_from_json(model_json)
    # load weights into loaded model
    loaded_model.load_weights('model.h5')
    print('Model loaded')

    # Evaluate model
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loss, accuracy = loaded_model.evaluate(test_data, hotone_test_labels, verbose=1)
    print('Loaded model. Test loss: ', loss, '\tTest accuracy: ', accuracy)
    """

    # https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3




######**********#######  MAIN  #########**********########
if __name__ == '__main__':
    main()