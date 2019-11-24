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
		
		It = img2 - img1
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
		
		indexes = np.ones((len(self.startXs)))
		n_values=0
		for i in range(len(self.startXs)):
			x = self.startXs[i]
			y = self.startYs[i]
			error=100
			#for j in range(Iterations):
			iterations=0
			while(error>0.0) and iterations<6:	
			
				try:
					#It_small = It[y-4:y+6,x-4:x+6]
					x_interp = np.arange(y-9,y+10)
					y_interp = np.arange(x-9,x+10)
					
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
					#b = np.matmul(np.reshape(ix,(1,len(ix))),np.reshape(iy,(len(iy),1)))
					b=np.sum(ix*iy)
					c = np.square(iy)
					c = np.sum(c)
					#e = np.matmul(np.reshape(ix,(1,len(ix))),np.reshape(it,(len(it),1)))
					e = np.sum(ix*it)
					#f = np.matmul(np.reshape(iy,(1,len(iy))),np.reshape(it,(len(it),1)))
					f = np.sum(iy*it)
					A = np.array([[a,b],[b,c]])
					#pdb.set_trace()
					B = np.array([e,f])
					B = np.reshape(B,(2,1))
					u_v = np.matmul(np.linalg.inv(A),B)
					x = x + u_v[0]
					y = y + u_v[1]
					#pdb.set_trace()
					A_error = np.hstack((np.reshape(ix,(len(ix),1)),np.reshape(iy,(len(iy),1))))
					b_error = np.reshape(it,(len(it),1))
					error = np.linalg.norm(np.matmul(A_error,u_v)+b_error,ord=2)
					#error = np.linalg.norm(u_v,ord=2)
					#print(error)
					iterations+=1
					#if iterations%100==0:
				#		print(iterations,x,y,u_v)
				except:
					print("failed")

			if (x <= width) and (y<=length) and error<0.10:
				self.newXs.append(x)
				self.newYs.append(y)
				n_values+=1
				#print("[IN Range]")
			else:
				#print("x is out of range for this case")
				indexes[i]=0
				
		#pdb.set_trace()
		self.startXs = self.startXs[indexes==1]
		self.startYs = self.startYs[indexes==1]
		self.newXs = np.array(self.newXs)
		self.newYs = np.array(self.newYs)
		return self.newXs, self.newYs

	def applyGeometricTransformation(self, bbox):
		'''This is to predict new bounding box coordinates. 
		Input bbox = np.array([x,y],[x,y],[x+w,y+h],[x+w,y+h])'''
		
		length = len(self.startXs)
		src = np.hstack((self.startXs.reshape(length,1),self.startYs.reshape(length,1)))
		dst = np.hstack((self.newXs.reshape(length,1),self.newYs.reshape(length,1)))
		#pdb.set_trace()
		tform = tf.estimate_transform('similarity', src, dst)
		center_x = bbox[0]+bbox[2]/2
		center_y = bbox[1]+bbox[3]/2
		old_coord = np.array([center_x,center_y,1]).reshape(3,1)
		new_coord = np.matmul(tform.params,old_coord)
		diff = np.linalg.norm(new_coord-old_coord)
		print("[values]",diff)
		if diff>30 :
			new_coord = old_coord.copy()
		#new_center = warp(np.array([center_x,center_y]),tform)
		
		self.new_bbox = np.zeros((4,1))
		self.new_bbox[0]= new_coord[0]-bbox[2]/2
		self.new_bbox[1]= new_coord[1]-bbox[3]/2
		self.new_bbox[2]= bbox[2]
		self.new_bbox[3]= bbox[3]

		return self.new_bbox

	















