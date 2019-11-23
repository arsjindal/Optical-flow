import pdb

from harris_corners import Harris_corners
from estimateAllTransitions import estimateTranslation
import bbox_create
import cv2
import numpy as np
from PIL import Image	

verbose=False

if __name__ == '__main__':
	image_1 = cv2.imread('Easy/0.png')
	image_2 = cv2.imread('Easy/1.png')
	
	bbox = bbox_create.bounding_box()	
	bboxes = bbox.create_bbox(image_1)
	x,y,w,h = bboxes[0]
	img_req_1 = cv2.cvtColor(image_1[y:y+h,x:x+w], cv2.COLOR_RGB2GRAY)
	img_req_2 = cv2.cvtColor(image_2[y:y+h,x:x+w], cv2.COLOR_RGB2GRAY)
	
	H = Harris_corners(img_req_1)
	gaussian_filter = H.gaussian_filter(1)
	suppressed_image = H.corner_detector(0.04)
	points_y, points_x= H.adap_supp_class(num_features=100)
	
	if verbose==True:
		Image.fromarray((suppressed_image*255).astype(np.uint8)).show()

	Translation = estimateTranslation(points_x,points_y)
	Translation.estimateAllTranslation(img_req_1,img_req_2)
	pdb.set_trace()