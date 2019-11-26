import numpy as np	
import cv2

import os

import pdb

class frame_reader(object):
	"""docstring for video_reader"""
	def __init__(self, location):
		super(frame_reader, self).__init__()
		self.location = location
		self.cam = cv2.VideoCapture(location) 

	def save_video(self, folder_name):
		print('Current Extracting frames.')
		try:
			os.mkdir(folder_name)
			current_frame = 0
			while True:
				ret, frame = self.cam.read()
				frame = frame.T
				frame = np.reshape(frame,(width,height,ch))
				if ret:
					name_frame = str(current_frame)
					name = folder_name+'/%s'%name_frame+'.png'
					# print('Creating...' + name_frame)
					# pdb.set_trace()
					cv2.imwrite(name, frame)
					# print(np.shape(frame))
					current_frame = current_frame + 1

				else:
					break
		except:
			print("folder already there")
		
		self.cam.release()


if __name__ == '__main__':
	Easy = frame_reader('Easy.mp4')
	Easy.save_video('Easy')
