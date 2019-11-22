import cv2
import numpy as np	

import pdb

class bounding_box(object):
	"""docstring for bounding_box"""
	def __init__(self	):
		super(bounding_box, self).__init__()
		
	def create_bbox(self, image):
		img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		img_blur = cv2.blur(img_gray, (3,3))
		ret,thresh = cv2.threshold(img_blur,140,255,0)
		pdb.set_trace()
		im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
		areas = []
		for i in range(len(contours)):
			area = cv2.contourArea(contours[i])
			areas.append(area)
		# areas = np.array(areas)
		pos = areas.index(max(areas))
		cnt= contours[pos]
		print(len(contours))
		self.x,self.y,self.w,self.h = cv2.boundingRect(cnt)
		cv2.rectangle(image,(self.x,self.y),(self.x+self.w,self.y+self.h),(0,255,0),2)
		cv2.imshow('image',image)
		cv2.waitKey(0)
		return self.x,self.y,self.w,self.h


if __name__ == '__main__':
	bbox = bounding_box()
	image= cv2.imread('Easy/0.png')
	bbox.create_bbox(image)

		