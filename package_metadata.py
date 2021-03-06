import argparse
import pickle

from tools.metadata_tools import *


def main():
    ####### Parameter parsing #######
    parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
    parser.add_argument('--dir', dest='dir', help='Directory path to metadata files', required=True)
    parser.add_argument('--ext', dest='ext', help='Extension of metadata files. Use it as filter in target directory.', default='json')
    parser.add_argument('--pkl', dest='pkl', help='File path to save pickle file.', required=True)
    args = parser.parse_args()
    #################################

    target_dir = args.dir
    extension = args.ext
    out_file = args.pkl

    metadata_matrix = metadata_to_matrix(target_dir, extension)    # sum element aggregation + class label matrix
    print('Metadata structure shape: ', metadata_matrix.shape)
    print('Instance sample: ', metadata_matrix[0, :])

    # save metadata structure into pickle file
    with(open(out_file, 'wb')) as pkl:
        pickle.dump(metadata_matrix, pkl, protocol=1)    # serialize matrix into binary string
        print('Pickle file saved: ' + out_file)



######**********#######  MAIN  #########**********########
if __name__ == '__main__':
    main()