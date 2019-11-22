import numpy as np
import cv2
import pdb
from PIL import Image
import time

class adap_supp(object):
	def __init__(self, num_features, r_values, x, y):
		self.num_features = num_features
		self.r_values = r_values
		self.x = x
		self.y = y
		self.points_ = []

	def adap_supp(self):
		length = len(self.r_values)

		points = dict.fromkeys(['r_value','x_index','y_index','radius_value'])

		indexes = self.r_values.argsort()

		points['r_value']=self.r_values[indexes]
		points['x_index']=self.x[indexes]
		points['y_index']=self.y[indexes]
		points['radius_value']=np.ones(length)*999999999

		for point in range(length-1):
		  
		  index_bool = (points['r_value']<1.9*points['r_value'][point]) & \
		               (points['r_value']>points['r_value'][point])
		  
		  if np.sum(index_bool)<0:
		    continue
		  
		  radius = min( (points['x_index'][index_bool]-points['x_index'][point])**2+\
		                (points['y_index'][index_bool]-points['y_index'][point])**2 )
		  points['radius_value'][point]=radius
		  
		indexes = points['radius_value'].argsort()[-self.num_features:]
		points['r_value'] = points['r_value'][indexes]
		points['x_index'] = points['x_index'][indexes]
		points['y_index'] = points['y_index'][indexes]
		points['radius_value'] = points['radius_value'][indexes]
		self.points_ = points

		return self.points_
