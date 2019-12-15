import numpy as np 
import pandas as pd
import os
import cv2

import tensorflow as tf

class IdentifyFace():
	def __init__(self, weight_path):

		self._weight_path = os.path.join(weight_path+'/model_5e.h5')
		self.__model = None
		self.__graph1 = None
		self.__session1 = None
	
		print("[INFO] Preparing Model and loading weights...")
		self.buildModel()
		print("[INFO] Done")

	def buildModel(self):
		self.__graph1 = tf.Graph()
		with self.__graph1.as_default():
			self.__session1 = tf.compat.v1.Session()
			with self.__session1.as_default():
				loaded_model = tf.keras.models.load_model(self._weight_path)
				self.__model = loaded_model

	# Prediction from all three classifiers. 
	def predict(self, image_dict):
		print("[INFO] Predicting Gender, Ethnicity, Age and Emotion...")
		output = {}
		for face_id, image in image_dict.items():

			img = cv2.resize(image, (200,200))
			img = img.reshape((1,)+img.shape)

			with self.__graph1.as_default():
				with self.__session1.as_default():
					predictions = self.__model.predict(img)
					# print("[CHECKING] Prediction :", predictions)
					# print("[CHECKING]: ", np.argmax(predictions[0]))
					# print("[CHECKING]: ", np.argmax(predictions[1]))
					# print("[CHECKING]: ", np.argmax(predictions[2]))
					
					gender = self.__decodeGender(np.argmax(predictions[0]))
					ethnicity = self.__decodeEnthicity(np.argmax(predictions[1]))
					age = self.__decodeAge(np.argmax(predictions[2]))

			output[face_id] = [gender, ethnicity, age]

		print("[INFO] Done")
		return output


	def __decodeGender(self, gender):
		if gender==0:
			return 'male'
		else:
			return 'female'

	def __decodeEnthicity(self, eth):
		if eth==0:
			return 'white'
		elif eth==1:
			return 'black'
		elif eth==2:
			return 'asian'
		elif eth==3:
			return 'indian'
		else:
			return 'others(hipsanic, latino, middle_eastern)'

	def __decodeAge(self, age):
		if age==0:
			return 'age-0-4'
		elif age==1:
			return 'age-5-17'
		elif age==2:
			return 'age-18-23'
		elif age==3:
			return 'age-24-25'
		elif age==4:
			return 'age-26'
		elif age==5:
			return 'age-27-29'
		elif age==6:
			return 'age-30-33'
		elif age==7:
			return 'age-34-37'
		elif age==8:
			return 'age-38-45'
		elif age==9:
			return 'age-46-54'
		elif age==10:
			return 'age-55-64'
		else:
			return 'age-65+'