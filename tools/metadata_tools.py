from os import listdir
from os.path import splitext
from itertools import compress
import numpy as np

from tools.metadata import Metadata
from monumai.monument import Monument


# list metadata file in folder by type
def read_metadata_file_paths(directory, type='json'):
    files = [f for f in listdir(directory)]

    # filter files by extension
    if type is not None:
        extension = "." + type      # add point for checking extension
        filtered = [splitext(file)[1]==extension for file in files]     #select files
        files = list(compress(files, filtered))

    return files


# return aggregation element vector by sum
def metadata_to_aggregation_sum(directory, filename):
    metadata = Metadata(directory, filename)
    monument = Monument(metadata)

    return monument.aggregation_score_sum()


# read metadata files from directory, aggregate, and append into matrix
def metadata_to_matrix(directory, type):
    file_paths = read_metadata_file_paths(directory, type)      # metadata files of type

    # aggregate metadata and store in matrix
    num_inst = len(file_paths)
    inst_len = sum([len(x) for x in list(Monument.ELEMENT_DIC.values())])
    matrix = []
    for path in file_paths:
        aggregation = metadata_to_aggregation_sum(directory, path)      # create generic "metadata_to_aggregation" if different aggregate operations needed
        matrix.append(aggregation)
    matrix = np.array( np.reshape(matrix, (num_inst, inst_len)) )       # list type to numpy array-based structure

    return matrix