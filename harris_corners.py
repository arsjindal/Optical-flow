import numpy as np
import cv2
import pdb
from PIL import Image
import time
import anms
import bbox_create
import estimateAllTransitions
import os

from config import *
'''
Takes input:
  image: grayscale image[0 to 1]
  sobel_x: sobel filter to operate in x direction
  sobel_y: sobel filter to operate in y direction
  gauss: gaussian filter 
corner_detector:  
  lambda: thershold ususally 0.04
adap_supp:
  num_features: top total number of features you need to extract

Output:
  H: image with harrisconners <0.05*H.max() intensity set to zero
  points: points in the image with radius_value, r_value, x_index, y_index
'''


class Harris_corners():
  def __init__(self, image, rgb_image):
    self.image = np.float32(image/image.max())
    self.s_x = np.array([[-1,0,+1],[-2,0,+2],[-1,0,+1]])
    self.s_y = np.array([[-1,-2,-1],[0,0,0],[+1,+2,+1]])
    self.rgb_image = rgb_image

  def gaussian_filter(self,sigma):
    size = np.ceil(3*sigma*2)+1
    x,y = np.mgrid[-size//2+1:size//2+1,-size//2+1:size//2+1]
    kernal = np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    return kernal/kernal.sum()

  def corner_detector(self, s_lambda):
    self.gauss = self.gaussian_filter(1)
    Ix = cv2.filter2D(self.image,-1,self.s_x)
    Iy = cv2.filter2D(self.image,-1,self.s_y)
 
    Ix_2 = cv2.filter2D(Ix**2,-1,self.gauss)
    Iy_2 = cv2.filter2D(Iy**2,-1,self.gauss)
    Ixy = cv2.filter2D(Ix*Iy,-1,self.gauss)

    H = (Ix_2*Iy_2 - Ixy**2) - s_lambda*(Ix_2+Iy_2)**2
    H[H<0.05*H.max()]=0
    
    self.x,self.y = np.where(H>=0.05)
    self.r_values = H[self.x,self.y]
    #pdb.set_trace()
    return H

  def adap_supp(self,num_features):
    length = len(self.r_values)
    
    points = dict.fromkeys(['r_value','x_index','y_index','radius_value'])
    
    indexes = self.r_values.argsort()
    
    points['r_value']=self.r_values[indexes]
    points['x_index']=self.x[indexes]
    points['y_index']=self.y[indexes]
    points['radius_value']=np.ones(length)*999999999
    #pdb.set_trace()
    for point in range(length-1):
      
      index_bool = (points['r_value']<1.9*points['r_value'][point]) & \
                   (points['r_value']>points['r_value'][point])
      
      if np.sum(index_bool)<0:
        continue
      
      radius = min( (points['x_index'][index_bool]-points['x_index'][point])**2+\
                    (points['y_index'][index_bool]-points['y_index'][point])**2 )
      points['radius_value'][point]=radius
      
    indexes = points['radius_value'].argsort()[-num_features:]
    points['r_value'] = points['r_value'][indexes]
    points['x_index'] = points['x_index'][indexes]
    points['y_index'] = points['y_index'][indexes]
    points['radius_value'] = points['radius_value'][indexes]
    
    return points

  def  adap_supp_class(self,num_features):
    adap = anms.adap_supp(num_features,self.r_values,self.x,self.y)
    points = adap.adap_supp()
    self.points_x = points['x_index']
    self.points_y = points['y_index']

    return self.points_x, self.points_y

if __name__ == "__main__":
  sobel_x = np.array([[-1, 0, +1],[-2, 0, +2],[-1, 0, +1]])
  sobel_y = np.array([[-1, -2, -1],[0, 0, 0],[+1, +2, +1]])
  bbox = bbox_create.bounding_box()
  image = cv2.imread('Easy/0.png')
  bboxes = bbox.create_bbox(image)
  x,y,w,h = bboxes[0]
  img_req = image[y:y+h,x:x+w]
  #pdb.set_trace()
  # img_gray = cv2.cvtColor(img_req, cv2.COLOR_RGB2GRAY)
  # H = Harris_corners(img_gray,img_req)
  # gaussian_filter = H.gaussian_filter(1)
  # suppressed_image = H.corner_detector(0.04)
  # points_x, points_y = H.adap_supp_class(num_features=100)

  # now using function estimate translation
  no_frames = os.listdir('Easy')
  for i in range(len(no_frames)):
    if(i>0):
      image = cv2.imread('Easy/%s.png'%str(i))
    if(i<len(no_frames-1)):
      image2 = cv2.imread('Easy/%s.png'%str(i+1))

    if(i==0):
      img_req = image[y:y+h,x:x+w]
      img_req2 = image2[y:y+h,x:x+w]
    else:
      img_req = image[new_bbox[0][1]:y[1]]
      
    img_gray = cv2.cvtColor(img_req, cv2.COLOR_RGB2GRAY)
    H = Harris_corners(img_gray,img_req)
    gaussian_filter = H.gaussian_filter(1)
    suppressed_image = H.corner_detector(0.04)
    points_x, points_y = H.adap_supp_class(num_features=100)
    est = estimateAllTransitions.estimateTranslation(points_x,points_y)



  #print(points_x)


