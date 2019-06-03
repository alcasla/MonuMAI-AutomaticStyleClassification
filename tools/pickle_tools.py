import numpy as np
import pickle
from random import randint
from tensorflow import keras

# return data, hotone encode labels, and instance filename
def prepare_metadatada_from_pkl_file(filepath):
    with(open(filepath, 'rb')) as pkl:
        metadata = pickle.load(pkl)
        print('PICKLE FILE READED: ', filepath, metadata.shape)

    # load instance filenames
    names = metadata[:, -1]     # load last column
    metadata = metadata[:, :metadata.shape[1]-1]    # remove name colum
    # prepare the input data to be modeled
    labels = np.array(metadata[:, -1], dtype=np.float)      # select last column
    labels = labels.astype(np.int)      # convert class label to int
    data = np.array(metadata[:, :metadata.shape[1]-1], dtype=np.float)      # select all except last column

    # convert the label to one-hot encode
    num_classes = max(labels) + 1  # max class index plus one (start from 0)
    hotone_labels = keras.utils.to_categorical(labels, num_classes)

    print('TRAIN data:', data.shape)
    num_sample = randint(0, labels.shape[0] - 1)
    print('*****\tSample ', data[num_sample, :])
    print('Class:', labels[num_sample], hotone_labels[num_sample], 'Name:', names[num_sample], '*****')

    return names, data, hotone_labels