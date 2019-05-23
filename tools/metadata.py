from os.path import join, splitext, split
import os
import numpy as np
import json


class Metadata:

	def __init__(self, dir, name):
		self.__filepath = join(dir, name)		# path to metadata file
		self.__obj_classes = []					# object classes
		self.__obj_scores = []					# object scores associated to class array

	@property
	def filepath(self):
		return self.__filepath

	@property
	def object_classes(self):
		return self.__obj_classes

	@property
	def object_scores(self):
		return self.__obj_scores

	def write_metadata_json(boxes, scores, classes, category_index, outfolder, inimage):
		# dictionary to store and append proposals
		data = {}
		data['object'] = []

		for i in range(classes.shape[1]):
			data['object'].append({
				'bndbox': {'xmin':repr(boxes[0, i, 0]), 'ymin':repr(boxes[0, i, 1]), 'xmax':repr(boxes[0, i, 2]), 'ymax':repr(boxes[0, i, 3])},
				'score': repr(scores[0,i]),
				'class': category_index[classes[0, i]]['name']
			})

		# number of prediction per class insertion
		data['num_predictions'] = []
		for idx_class in np.unique(classes[0,:]):
			matches = sum(classes[0, :] == float(idx_class))
			data['num_predictions'].append({category_index[idx_class]['name']: str(matches)})

		# parse input path to get image name
		image_name = split(inimage)   # divide path into parts
		image_name = image_name[len(image_name)-1]  # select last = file name

		# output image path insertion
		data['image'] = join(outfolder, image_name)
		print(join(outfolder, image_name))

		# write date as json
		with (join(outfolder, splitext(image_name)[0])+'.json', 'w') as outfile:
			json.dump(data, outfile)

	# generic function to load metadata from json or xml file
	def load_metadata(self):
		ext = os.path.splitext(self.filepath)[1]		# metadata file extension
		if ext == '.json':
			self.load_metadata_json()
		if ext == '.xml':
			self.load_metadata_xml()

	def load_metadata_json(self):
		print('Load .json metadata: ', self.filepath)
		with open(self.filepath) as json_file:
			data = json.load(json_file)
			predictions = np.array(data.get('object', []))		# Get object prediction list

			# for each prediction save class and score
			for obj in range(len(predictions)):
				self.object_classes.append(predictions[obj]['class'])
				self.object_scores.append(np.float(predictions[obj]['score']))		# score np.float

	def load_metadata_xml(self):
		print('Load .xml metadata: ', self.filepath)
		print('FUNCIÓN NO TERMINADA')