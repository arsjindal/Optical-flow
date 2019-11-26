import numpy as np
import cv2


from PIL import Image
from scipy import interpolate

import pdb

WINDOW_SIZE = 25

def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2):
    x_iter = startX
    y_iter = startY
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    x_range = np.arange(Ix.shape[1])
    y_range = np.arange(Ix.shape[0])

    Img1_interp = interpolate.interp2d(x_range,y_range,img1_gray,kind='cubic')
    Img2_interp = interpolate.interp2d(x_range,y_range,img2_gray,kind='cubic')
    Ix_interp = interpolate.interp2d(x_range,y_range,Ix,kind='cubic')
    Iy_interp = interpolate.interp2d(x_range,y_range,Iy,kind='cubic')

    x_interp = np.arange(y_iter-12,y_iter+12)[:24]
    y_interp = np.arange(x_iter-12,x_iter+12)[:24]		

    Img2_small = Img2_interp(y_interp,x_interp)
    Img1_small = Img1_interp(y_interp,x_interp)
    It_small = Img1_small-Img2_small

    I1_value = Img1_small.flatten()
    Ix_small = Ix_interp(y_interp,x_interp)
    Ix_value = Ix_small.flatten()					
    Iy_small = Iy_interp(y_interp,x_interp)
    Iy_value = Iy_small.flatten()


    I = np.vstack((Ix_value,Iy_value))
    a = np.sum(np.square(Ix_value))
    b = np.sum(Ix_value*Iy_value)
    c = np.sum(np.square(Iy_value))
    A = np.array([[a,b],[b,c]])

    
    iteration=0
    while iteration<15:
        iteration = iteration+1
        x_interp = np.arange(y_iter-12,y_iter+12)[:24]
        y_interp = np.arange(x_iter-12,x_iter+12)[:24]	 
        I2_value = Img2_interp(y_interp,x_interp)   
        I2_value = I2_value.flatten()

        Ip = (I2_value-I1_value).reshape((-1,1))
        b = -I.dot(Ip)
        u_v = np.linalg.inv(A).dot(b)

        x_iter += u_v[0,0]
        y_iter += u_v[1,0]

    return x_iter, y_iter

