import cv2
import numpy as np
import pdb

bbox = []
class bounding_box(object):
  def __init__(self):
    super(bounding_box,self).__init__()

  def create_bbox(self, im):
    
    while True:
      r = cv2.selectROI(im)
      bbox.append(r) 
      print("[Selected Points]",bbox)
      print("Press q to quit or c to continue ")
      k = cv2.waitKey(0) & 0xFF
      if (k==113):
        break
    
    #pdb.set_trace()
    return bbox

if __name__ == '__main__' :
    image = cv2.imread('Easy/0.png')
    bbox_1 = bounding_box()
    bbox_1.create_bbox(image)
