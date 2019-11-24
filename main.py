import pdb

import glob

from harris_corners import Harris_corners
from estimateAllTransitions1 import estimateTranslation
import bbox_create
import cv2
import numpy as np
from PIL import Image	
from scipy import signal

verbose=False
image_files = glob.glob("Easy/*.png")
image_files.sort()
# print(image_files)

if __name__ == '__main__':
	image_1 = cv2.imread('Easy/000.png')
	image_2 = cv2.imread('Easy/001.png')
	
	bbox = bbox_create.bounding_box()	
	bboxes = bbox.create_bbox(image_1)
	new_bbox = bboxes[0]
	#pdb.set_trace()
	#for i in range(len(image_files)-1):
	for i in range(300):
		x = int(new_bbox[0])
		y = int(new_bbox[1])
		w = int(new_bbox[2])
		h = int(new_bbox[3])
		
		image_1 = cv2.imread(image_files[i])
		image_2 = cv2.imread(image_files[i+1])
		img_req_1 = cv2.cvtColor(image_1[y:y+h,x:x+w], cv2.COLOR_RGB2GRAY)
		img_req_2 = cv2.cvtColor(image_2[y:y+h,x:x+w], cv2.COLOR_RGB2GRAY)
		img_req_1 = img_req_1/255
		img_req_2 = img_req_2/255
		H = Harris_corners(img_req_1)
		gaussian_filter = H.gaussian_filter(0.5)
		suppressed_image = H.corner_detector(0.04)
		points_y, points_x= H.adap_supp_class(num_features=100)
		# converting points_y and points_x into absolute coordinates

		img_req_1 = signal.convolve2d(img_req_1,gaussian_filter,boundary='symm',mode='same')
		img_req_2 = signal.convolve2d(img_req_2,gaussian_filter,boundary='symm',mode='same')
		#corners = cv2.goodFeaturesToTrack(img_req_1,25,0.01,10)
		#corners = np.int0(corners.squeeze())
		#points_x = corners[:,0]
		#points_y = corners[:,1]
		if verbose==True:
			Image.fromarray((suppressed_image*255).astype(np.uint8)).show()

		Translation = estimateTranslation(points_x,points_y)
		Translation.estimateAllTranslation(img_req_1,img_req_2)
		new_bbox = Translation.applyGeometricTransformations(new_bbox,i)
		
		new_bbox = new_bbox.astype(np.uint)
		start_point = (new_bbox[0],new_bbox[1])
		end_point = (new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3])
		image_1 = cv2.rectangle(image_1,start_point,end_point,(255,0,255),5)
		if (i%10==0):
			Image.fromarray(image_1).show()
		#pdb.set_trace()
		#print(i)
		#print(new_bbox)
	# pdb.set_trace()