import cv2
import numpy as np	
from skimage import transform as tf

class applyGeometricTransformations(object):
	"""docstring for applyGeometricTransformations"""
	def __init__(self, startXs, startYs, newXs, newYs, bbox):
		super(applyGeometricTransformations, self).__init__()
		self.startXs = startXs
		self.startYs = startYs
		self.newXs = newXs
		self.newYs = newYs
		self.bbox = bbox

	def transform(self):
		no_objects = self.bbox.shape[0]
		self.newbbox = np.zeros_like(self.bbox)
		self.Xs = self.newXs.copy()
		self.Ys = self.newYs.copy()
		for idx in range(no_objects):
			startXs_obj = self.startXs[:,[idx]]
			startYs_obj = self.startYs[:,[idx]]
			newXs_obj = self.newXs[:,[idx]]
			newYs_obj = self.newYs[:,[idx]]
			src = np.hstack((startXs_obj,startYs_obj))
			dst = np.hstack((newXs_obj,newYs_obj))
			tform = tf.SimilarityTransform()
			tform.estimate(dst=dst,src=src)
			matrix = tform.params
			threshold = 1
			projected_points = matrix.dot(np.vstack((src.T.astype(float),np.ones([1,np.shape(src)[0]]))))
			distance = np.square(projected_points[0:2,:].T - dst).sum(axis=1)
			dst_inliners = dst[distance<threshold]
			src_inliners = src[distance<threshold]
			if np.shape(src_inliners)[0]<4:
				print('Very less points')
				dst_inliners = dst
				src_inliners = src
			tform.estimate(dst=dst_inliners, src=src_inliners)
			matrix = tform.params
			coords = np.vstack((self.bbox[idx,:,:].T,np.array([1,1,1,1])))
			new_coord = matrix.dot(coords)
			self.newbbox[idx,:,:] = new_coord[0:2,:].T
			self.Xs[distance>=threshold,idx]= -1
			self.Ys[distance>=threshold,idx]= -1

		return self.Xs,self.Ys, self.newbbox
		