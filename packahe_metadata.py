import argparse

from tools.metadata_tools import *


def main():
    ####### Parameter parsing #######
    parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
    parser.add_argument('--dir', dest='dir', help='Directory path to metadata files', required=True)
    parser.add_argument('--ext', dest='ext', help='Extension of metadata files. Use it as filter in target directory.')
    args = parser.parse_args()
    #################################

    metadata_matrix = metadata_to_matrix(args.dir, args.ext)    # sum element aggregation and store in matrix metadata files
    print( metadata_matrix.shape )

    # funciones para salvar matrices en ficheros pickle
        #leer metadatos desde el etiquetado xml



######**********#######  MAIN  #########**********########
if __name__ == '__main__':
    main()