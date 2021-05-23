#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import math

class GLCM:

    #Define the maximum number of gray levels
    gray_level = 16

    def maxGrayLevel(self, img):
        max_gray_level=0
        (height,width)=img.shape
        for y in range(height):
            for x in range(width):
                if img[y][x] > max_gray_level:
                    max_gray_level = img[y][x]
        return max_gray_level+1

    def getGlcm(self, input,d_x,d_y):
        srcdata=input.copy()
        ret=[[0.0 for i in range(self.gray_level)] for j in range(self.gray_level)]
        (height,width) = input.shape

        max_gray_level=self.maxGrayLevel(input)

        #If the number of gray levels is greater than gray_level, reduce the gray level of the image to gray_level and reduce the size of the gray level co-occurrence matrix
        if max_gray_level > self.gray_level:
            for j in range(height):
                for i in range(width):
                    srcdata[j][i] = srcdata[j][i]*self.gray_level / max_gray_level

        for j in range(height-d_y):
            for i in range(width-d_x):
                 rows = srcdata[j][i]
                 cols = srcdata[j + d_y][i+d_x]
                 ret[rows][cols]+=1.0

        for i in range(self.gray_level):
            for j in range(self.gray_level):
                ret[i][j]/=float(height*width)

        return ret

    def feature_computer(self, p):
        Con=0.0
        Eng=0.0
        Asm=0.0
        Idm=0.0
        for i in range(self.gray_level):
            for j in range(self.gray_level):
                Con+=(i-j)*(i-j)*p[i][j]
                Asm+=p[i][j]*p[i][j]
                Idm+=p[i][j]/(1+(i-j)*(i-j))
                if p[i][j]>0.0:
                    Eng+=p[i][j]*math.log(p[i][j])
        return Asm,Con,-Eng,Idm

    def test(self, img):
        try:
            img_shape=img.shape
        except:
            print ('imread error')
            return -1

        img=cv2.resize(img,(int(img_shape[1]/2),int(img_shape[0]/2)),interpolation=cv2.INTER_CUBIC)

        img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        glcm_0=self.getGlcm(img_gray, 1,0)
        #glcm_1=getGlcm(src_gray, 0,1)
        #glcm_2=getGlcm(src_gray, 1,1)
        #glcm_3=getGlcm(src_gray, -1,1)

        asm,con,eng,idm=self.feature_computer(glcm_0)

        return (asm,con,eng,idm)

if __name__=='__main__':
    img = cv2.imread("C:\\Users\Yasiru\Documents\Projects\diabetic-retinopathy-detection\Classifier\\test\\test-img.jpg")
    gl = GLCM()
    data = gl.test(img)
    print(data)