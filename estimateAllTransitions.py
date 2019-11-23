import numpy as np
import cv2
from scipy import signal
from skimage import transform as tf
import pdb
from scipy import interpolate

class estimateTranslation():
	"""docstring for estimateAllTranslation"""
	def __init__(self, startXs, startYs):
		#super(estimateTranslation, self).__init__()
		self.startXs = startXs
		self.startYs = startYs
		self.newXs = []
		self.newYs = []
	
	def estimateAllTranslation(self, img1, img2):
		'''I am going to create an array A=[[a,b],[b,c]], B = [[e],[f]] 
		& the final equation will result in A[[u],[v]]= B. 
		Do remember x represents column & y= row. 
		I am only making 5 iteration to get more accurate results. this can be changed 
		by changing value of variable Iterations''' 
		print("here")
		It = img2 - img1
		#pdb.set_trace()
		length, width = np.shape(img2)
		sobel_x = np.array([[-1, 0, +1],[-2, 0, +2],[-1, 0, +1]])
		sobel_y = np.array([[-1, -2, -1],[0, 0, 0],[+1, +2, +1]])
		Ix = signal.convolve2d(img1,sobel_x, boundary='symm', mode='same')
		Iy = signal.convolve2d(img2,sobel_y, boundary='symm', mode='same')
		x_range = np.arange(Ix.shape[1])
		y_range = np.arange(Ix.shape[0])
		It_interp = interpolate.interp2d(x_range,y_range,It,kind='cubic')
		Ix_interp = interpolate.interp2d(x_range,y_range,Ix,kind='cubic')
		Iy_interp = interpolate.interp2d(x_range,y_range,Iy,kind='cubic')
		#pdb.set_trace()
		Iterations = 5
		for i in range(len(self.startXs)):
			x = self.startXs[i]
			y = self.startYs[i]
			for j in range(Iterations):
				
				#It_small = It[y-4:y+6,x-4:x+6]
				x_interp = np.arange(y-4,y+6)
				y_interp = np.arange(x-4,x+6)
				It_small = It_interp(y_interp,x_interp)
				it = It_small.flatten()
				Ix_small = Ix_interp(y_interp,x_interp)
				#Ix_small = Ix[y-4:y+6,x-4:x+6]
				ix = Ix_small.flatten()
				#Iy_small = Iy[y-4:y+6,x-4:x+6]
				Iy_small = Iy_interp(y_interp,x_interp)
				iy = Iy_small.flatten()
				a = np.square(ix)
				a = np.sum(a)	
				b = np.matmul(np.reshape(ix,(1,len(ix))),np.reshape(iy,(len(iy),1)))
				c = np.square(iy)
				c = np.sum(c)
				e = np.matmul(np.reshape(ix,(1,len(ix))),np.reshape(it,(len(it),1)))
				f = np.matmul(np.reshape(iy,(1,len(iy))),np.reshape(it,(len(it),1)))
				A = np.array([[a,b[0,0]],[b[0,0],c]])
				B = np.array([e,f])
				B = np.reshape(B,(2,1))
				u_v = np.matmul(np.linalg.inv(A),B)
				x = x + u_v[0]
				y = y + u_v[1]
				#pdb.set_trace()
			if (x <= width):
				self.newXs.append(x)
			else:
				print("x is out of range for this case")
			if (y <= length):
				self.newYs.append(y)
			else:
				print("y is out of range for this case")
			pdb.set_trace()
		return self.newXs, self.newYs

	def applyGeometricTransformation(self, bbox):
		'''This is to predict new bounding box coordinates. 
		Input bbox = np.array([x,y],[x,y],[x+w,y+h],[x+w,y+h])'''
		src = np.array([np.array(self.startXs),np.array(self.startYs)])
		dst = np.array([np.array(self.newXs), np.array(self.newYs)])
		tform = tf.estimate_transform('similarity', src, dst)
		self.new_bbox = warp(bbox,tform)

		return self.new_bbox

	














