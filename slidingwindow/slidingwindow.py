# -*- coding: utf-8 -*-

import cv2
import pylab as pl
import numpy as np
#import pandas as pd 

image_path = 'C:/Users/Nantan/Documents/mediaseek/OCR/data/Sample_URL_Images/URL1/freefont_logo_ume-pmo3.png'



# urlのある範囲を見つける
def find_url(image):
    height, width = image.shape
    image_blur = cv2.medianBlur(image, 5)
    thres = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)
    
    #image = pd.DataFrame(image)
    #image.to_csv('a.csv')
    _ret, contours, _hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours)) 
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(1, len(contours)):
        #cv2.drawContours(thres, contours[i], -1, (255, 0, 0), 3)
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])
    x1_min = min(x1)
    y1_min = min(y1)
    x2_max = max(x2)
    y2_max = max(y2)
    return thres, x1_min, y1_min, x2_max, y2_max
    """
    # 枠取りをした結果を表示
    cv2.rectangle(thres, (x1_min, y1_min), (x2_max, y2_max), (255, 255, 255), 3)
    #cv2.imwrite('cropped_edge_rectangle.jpg', thres)
    
    cv2.imshow('contours', thres)
    cv2.imshow('original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

def sliding_window(image_path, stepSize):
    image = cv2.imread(image_path, 0)
    thres, x1_min, y1_min, x2_max, y2_max = find_url(image)
    windowsize = int((y2_max - y1_min) / 4)
    x1 = x1_min
    x2 = x1_min + windowsize
    i = 0
    while x2 <= x2_max:
        #cv2.rectangle(image, (x1, y1_min), (x2, y2_max), (255, 0, 0), 1)
        cv2.imwrite('images/window'+str(i)+'.jpg', image[y1_min:y2_max+1, x1:x2+1])
        x1 += stepSize
        x2 += stepSize
        i += 1
    cv2.imwrite('images/window'+str(i)+'.jpg', image[y1_min:y2_max+1, x2_max-windowsize:x2_max+1])
    #cv2.rectangle(image, (x2_max-windowsize, y1_min), (x2_max, y2_max), (255, 0, 0), 1)
    #cv2.imshow('contours', thres)
    """
    cv2.imshow('original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    """
    pl.gray()
    pl.matshow(image)
    pl.show
    """
    """
    for y in range(0, width, 2):
        break
        clp1 = np.sum(image[0:height, y:y+1])
        clp2 = np.sum(image[0:height, y+1:y+2])
        ratio_dense = clp2 / clp1
        if ratio_dense < 0.8:
            pl.gray()
            pl.matshow(image[0:height, y+1:width])
            pl.show
            break
    """
    


sliding_window(image_path, 20)