import argparse
from tensorflow import keras
from tensorflow.python.keras.models import model_from_json

from tools.pickle_tools import *


def main():
    ####### Parameter parsing #######
    parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
    parser.add_argument('--mdl', dest='model', help='File path to load saved model.', required=True)
    parser.add_argument('--tst', dest='test', help='File path to load test pickle file.', required=True)

    args = parser.parse_args()
    #################################

    modelfile = args.model
    test = args.test
    print( modelfile )
    print( test )

    ############## Data preparation ##############
    # load test metadata structure from pickle file
    filenames, test_data, test_label = prepare_metadatada_from_pkl_file(test)

    ############# Model  ###############
    json = open(modelfile+'.json', 'r')
    model_json = json.read()
    json.close()
    loaded_model = model_from_json(model_json)
    # load weights into loaded model
    loaded_model.load_weights(modelfile+'.h5')
    print('Model loaded')

    # Evaluate model
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loss, accuracy = loaded_model.evaluate(test_data, test_label, verbose=1)
    print('Loaded model. Test loss: ', loss, '\tTest accuracy: ', accuracy)

    ######### Prediction information #########
    predict_class = loaded_model.predict_classes(test_data, batch_size=1)
    predict_probs = loaded_model.predict(test_data, batch_size=1)       # for each instance predict probability for each class

    # compare classification against label truth
    right_clas = [test_label[indx, predict_class[indx]]==1.0 for indx in range(predict_class.shape[0])]     # look in the predicted position of the hotone encode, 1.0 = success
    success_indx = np.where(right_clas)[0]
    mistake_indx = np.where(np.logical_not(right_clas))[0]     # reverse boolean values to get mistakes

    # successful classifications
    print('Successful classification: ', len(success_indx))
    for indx in success_indx:
        print('\t', filenames[indx])
        print('\t', predict_probs[indx])

    # wrong classifications
    print('Wrong classification: ', len(mistake_indx))
    for indx in mistake_indx:
        print('\t', filenames[indx])
        print('\t', predict_probs[indx])



######**********#######  MAIN  #########**********########
if __name__ == '__main__':
    main()