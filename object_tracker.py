import cv2
import numpy as np	
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformations import applyGeometricTransformations
import frame_extractor
import os
import argparse

class object_tracker(object):
	"""docstring for object_tracker"""
	def __init__(self, video):
		super(object_tracker, self).__init__()
		self.video = video
		self.Easy = frame_extractor.frame_reader(self.video)

	def collecting_no_of_frames(self,folder_name):
		self.folder_name = folder_name
		if not os.path.exists(self.folder_name):
			os.mkdir(self.folder_name)
		self.Easy.save_video(self.folder_name)
		self.names = os.listdir(self.folder_name)
		self.names.sort()
		self.no_frames = len(self.names)
		self.bbox = np.empty((self.no_frames,), dtype=np.ndarray)
		self.center = np.empty((self.no_frames,), dtype=np.ndarray)
		self.center2 = np.empty((self.no_frames,), dtype=np.ndarray)
		# self.frames_draw = np.empty((self.no_frames,), dtype=np.ndarray)
		

	def draw_bounding_box(self):
		self.no_object = int(input("Enter number of bjects to track"))
		self.bbox[0] = np.empty((self.no_object,4,2), dtype=float)
		self.center[0]= np.empty((self.no_object,1,2), dtype=int)
		self.center2[0]= np.empty((self.no_object,1,2), dtype=int)
		self.img1 = cv2.imread(self.folder_name+'/%s.png'%str(0))
		for i in range(self.no_object):
			x,y,w,h = cv2.selectROI("Select Object %d"%(i),self.img1)
			cv2.destroyWindow("Select Object %d"%i)
			self.bbox[0][i,:,:]= np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]]).astype(float)
			# self.center[0][i,:,:]= np.array([x+w/2,y+h/2])

	def track_object(self):
		template = cv2.cvtColor(self.img1,cv2.COLOR_BGR2GRAY)
		out = cv2.VideoWriter('Results/'+self.folder_name+'_results.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,(template.shape[1],template.shape[0]))
		self.startXs, self.startYs = getFeatures(template,self.bbox[0],use_shi=False)
		for i in range(1,self.no_frames):
			print('Tracking for frame: %s'%str(i))
			self.frame1 = cv2.imread(self.folder_name+'/%s.png'%str(i-1))
			self.frame2 = cv2.imread(self.folder_name+'/%s.png'%str(i))
			newXs, newYs = estimateAllTranslation(self.startXs,self.startYs, self.frame1, self.frame2)
			ag = applyGeometricTransformations(self.startXs, self.startYs, newXs, newYs, self.bbox[i-1])
			Xs, Ys, self.bbox[i] = ag.transform()
			self.center[i]= np.empty((self.no_object,1,2), dtype=int)
			self.center2[i]= np.empty((self.no_object,1,2), dtype=int)

			# updating coordinates
			self.startXs = Xs
			self.startYs = Ys

			# updating feature points
			no_features_left = np.sum(Xs!=-1)
			print('No. of features: %d'%no_features_left)
			if no_features_left<15:
				print('Generate new features')
				self.startXs, self.startYs = getFeatures(cv2.cvtColor(self.frame2,cv2.COLOR_BGR2GRAY),self.bbox[i])

			# drawing bounding box and visualising feature point for each object
			frames_draw = self.frame2.copy()
			
			for j in range(self.no_object):
				x,y,w,h = cv2.boundingRect(self.bbox[i][j,:,:].astype(int))
				self.center[i][j,:,:]= np.array([x,y])
				self.center2[i][j,:,:]= np.array([x,y+h])
				frames_draw = cv2.rectangle(frames_draw, (x,y),(x+w,y+h), (255,0,0), 2)
				for k in range(self.startXs.shape[0]):
					frames_draw= cv2.circle(frames_draw, (int(self.startXs[k,j]),int(self.startYs[k,j])),3,(0,255,0),thickness=2)
				center_points1=[]
				center_points2=[]
				for l in range(i+1):
					center_points1.append(self.center[l][j,:,:])
					center_points2.append(self.center2[l][j,:,:])
				if(j==1):
					frames_draw = cv2.drawContours(frames_draw, np.array(center_points1), -1, (0,0,255), 3)
					frames_draw = cv2.drawContours(frames_draw, np.array(center_points2), -1, (0,0,255), 3)
				else:
					frames_draw = cv2.drawContours(frames_draw, np.array(center_points1), -1, (255,255,255), 3)
					frames_draw = cv2.drawContours(frames_draw, np.array(center_points2), -1, (255,255,255), 3)
			cv2.imshow("window", frames_draw)
			cv2.waitKey(10)
			out.write(frames_draw)

		out.release()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Videos to track Object')
	parser.add_argument('indir', type=str, help='Input dir for video')
	parser.add_argument('folder', type=str, help='Folder for saving extracted Frames')
	args = parser.parse_args()
	# parser.add_argument('outdir', type=str, help='Folder for saving extracted Frames')
	# ot = object_tracker('Videos/Easy.mp4')
	ot = object_tracker(args.indir)
	ot.collecting_no_of_frames(args.folder)
	ot.draw_bounding_box()
	ot.track_object()

	# applying for medium 
	# ot_m = object_tracker('Videos/Medium.mp4')
	# ot_m.collecting_no_of_frames('Medium')
	# ot_m.draw_bounding_box()
	# ot_m.track_object()












		
