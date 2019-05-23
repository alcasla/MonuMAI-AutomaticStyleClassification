import numpy as np

class Monument:
    ELEMENT_DIC = {
        'hispanic-muslim': ['arco-herradura',   'arco-lobulado',   'dintel-adovelado'],
        'gothic':       ['arco-apuntado',   'arco-conopial',   'arco-trilobulado'],
        'renaissance':  ['arco-medio-punto',   'vano-adintelado',   'ojo-de-buey',   'fronton',   'fronton-curvo',   'serliana'],
        'baroque':      ['arco-medio-punto',   'vano-adintelado',    'ojo-de-buey',   'fronton-partido',   'columna-salomonica']
    }

    def __init__(self, metadata):
        self.__elements = {}        # dictipnary to group objects by elements
        self.__aggregation = {}
        self.__metadata = metadata
        self.__upload_metadata()

    # load metadata, import and order elements into a dictionary
    def __upload_metadata(self):
        self.__metadata.load_metadata()     # load metadata into the object
        styles = self.ELEMENT_DIC.keys()
        for stl in styles:      # each style
            self.__elements[stl] = {}      # insert style key
            elems = self.ELEMENT_DIC[stl]
            for e in elems:         # each element
                scores_e = self.__get_element_scores(e)     # scores of the element
                self.__elements[stl][e] = scores_e          # save scores in dictionary style-elements

    def __get_element_scores(self, element):
        # get index of element and return their scores
        elem_indx = [i for i,x in enumerate(self.__metadata.object_classes) if x==element]      # indexes of element
        scores = np.asarray(self.__metadata.object_scores, dtype=np.float)
        return scores[elem_indx]

    # return array contains the score aggregation
    def aggregation_score_sum(self):
        self.__aggregate_scores_sum()       # perform the element aggregation
        aggregation = np.array([])      # vector to store aggregated scores
        # insert in an array each element aggregated score
        for stl in self.__aggregation.keys():
            for e in self.__aggregation[stl]:
                aggregation = np.append(aggregation, float(self.__aggregation[stl][e]))
        return aggregation

    # generate the aggregation dictionary of elements with the score sum
    def __aggregate_scores_sum(self):
        styles = self.__elements.keys()
        for stl in styles:      # each style
            self.__aggregation[stl] = {}  # insert style key
            elems = self.__elements[stl]
            for e in elems:         # each element
                tot_score = np.sum(self.__elements[stl][e])     # aggregate scores
                self.__aggregation[stl][e] = tot_score