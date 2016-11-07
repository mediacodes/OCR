# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:52:55 2016

@author: 武本
"""

#coding:utf-8
import numpy as np
import time as time
import sys
import pylab as pl
import pandas as pd
#from mlp import MultiLayerPerceptron
#from sklearn.datasets import fetch_mldata
#from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import LabelBinarizer
#from sklearn.metrics import confusion_matrix, classification_report
#from matplotlib import pyplot as plt
#from matplotlib import cm
from scipy.misc import imread,imresize
#import csv
import cv2
from sklearn.externals import joblib
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
#from scipy.misc import imresize

""""""""""""""""""""""""""" variables """""""""""""""""""""""""""""""""
image_path ='C:\\Users\\mech-user\\Documents\\Tesseract\\pictures\\ruby.png'
learning_model_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\Learning_Models\\SVC(changed13)_resized(20)'
csv_result_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\results\\find_letter_in_pic\\camera\\ruby_result.csv'
csv_string_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\results\\find_letter_in_pic\\camera\\ruby_string.csv'


 
 
""""""""""""""""""""""""""" functions """""""""""""""""""""""""""""""""
#初期パラメータ
beta = 5 #画素値の連続をどのくらい重視するか
eta = 0.1 #ノイズいり画像とどれくら類似させるか
h = 0.01 #黒の画素をどれくらい残すか
eps = 0.000001 #収束域

""" 全エネルギー計算 """
def E(xs, ys, width, height):
    S = width * height
    s0 = s1 = s2 = 0
    for i in range(S):
        s0 += xs[i]
        s2 += xs[i] * ys[i]
    #右、下隣のみを想定
    #右、下端の例外処理は考えてない
    for y in range(height-3):
        for x in range(width-3):
            i = y * width + x
            #s1 += xs[i] * xs[i+1] + xs[i] * xs[i+width] #1ピクセル
            #s1 += xs[i] * xs[i+1] + xs[i] * xs[i+width] + xs[i] * xs[i+2] + xs[i] * xs[i+2*width] #2ピクセル
            s1 += xs[i] * xs[i+1] + xs[i] * xs[i+width] + xs[i] * xs[i+2] + xs[i] * xs[i+2*width] + xs[i] * xs[i+3] + xs[i] * xs[i+3*width] #3ピクセル
            
    return h * s0 - beta * s1 -eta *s2

""" ノイズ除去 """
def remove_noise(img):
    width, height = img.shape[1], img.shape[0]
    S = img.size #全画素数
    xs = [0] * S #復元したい画像の配列を用意
    ys = [0] * S #ノイズ入り画像の配列を用意(二値化用)
    #初期値代入(復元した画像とノイズ入り画像を同じにする)
    for y in range(height-1):
        for x in range(width-1):
            i = y * width + x
            xs[i] = ys[i] = 1 if img[y,x] == 255 else -1
    
    #一つの画素に対して反転前後のエネルギーを比較する
    def de(i):
        s0 = xs[i]
        s1 = 0
        #ここも大体でやってる気がする
        if i > 0:         s1 += xs[i] * xs[i-1] #1ピクセル
        if i > 1:         s1 += xs[i] * xs[i-2] #2ピクセル
        if i > 2:         s1 += xs[i] * xs[i-3] #3ピクセル
        if i < S-1 :      s1 += xs[i] * xs[i+1] #1ピクセル
        if i < S-2 :      s1 += xs[i] * xs[i+2] #2ピクセル
        if i < S-3 :      s1 += xs[i] * xs[i+3] #3ピクセル
        if i >= width:    s1 += xs[i] * xs[i-width] #1ピクセル
        if i >= 2*width:  s1 += xs[i] * xs[i-2*width] #2ピクセル
        if i >= 3*width:  s1 += xs[i] * xs[i-3*width] #3ピクセル
        if i < S-width:   s1 += xs[i] * xs[i+width] #1ピクセル
        if i < S-2*width: s1 += xs[i] * xs[i+2*width] #2ピクセル　
        if i < S-3*width: s1 += xs[i] * xs[i+3*width] #3ピクセル
        """ここの部分は無視,文字も消してしまう
        if i > 0 and i >= width: s1 += xs[i] * xs[i-width-1] + xs[i] * xs[i-width+1]
        if i < S-1 and i < S-width: s1 += xs[i] * xs[i+width-1] + xs[i] * xs[i-width+1]
        """
        s2 = xs[i] * ys[i]
        curr_e = h * s0 - beta * s1 - eta * s2
        toggled_e = -curr_e

        return toggled_e < curr_e
    
    #1,-1の二値化を画素数255,0に戻す
    def reflect():
        for i in range(S):
            x = i % width
            y = i // width
            img[y,x] = 255 if xs[i] == 1 else 0
            
    energy = E(xs, ys, width, height)
    
    #読み取りを10回まで繰り返す
    for j in range(2):
        #print(j)
        for i in range(S):
            if de(i): xs[i] = -xs[i]
        new_energy = E(xs, ys, width, height)
        if energy - new_energy < eps: break #収束域に入ると終了する
        energy = new_energy
        
    reflect()
    return img
    
""" 4点を指定してトリミングする。 """
def transform_by4(img, points):
    top = points[:2]
    bottom = points[2:]
    points = np.array(top + bottom, dtype='float32')  # 分離した二つを再結合。
    
#    width = max(np.sqrt(((points[0][0]-points[2][0])**2)*2), np.sqrt(((points[1][0]-points[3][0])**2)*2))
#    height = max(np.sqrt(((points[0][1]-points[2][1])**2)*2), np.sqrt(((points[1][1]-points[3][1])**2)*2))
    
    width = max(np.sqrt(((points[0][0]-points[1][0])**2)+((points[0][1]-points[1][1])**2)), np.sqrt(((points[2][0]-points[3][0])**2)+((points[2][1]-points[3][1])**2)))
    height = max(np.sqrt(((points[0][0]-points[3][0])**2)+((points[0][1]-points[3][1])**2)), np.sqrt(((points[1][0]-points[2][0])**2)+((points[1][1]-points[2][1])**2)))


    dst = np.array([np.array([0, 0]),
                    np.array([width-1, 0]),
                    np.array([width-1, height-1]),
                    np.array([0, height-1]),], np.float32)
    
    trans = cv2.getPerspectiveTransform(points, dst)  # 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。
    return cv2.warpPerspective(img, trans, (int(width), int(height)))  # 透視変換行列を使って切り抜く。
    
       
        
""" 本プログラム """
if __name__ == '__main__':
    """ obtain image """#
    img = cv2.imread(image_path,0)
    if len(img.shape) == 3:
        img_height, img_width, img_channels = img.shape[:3]
    else:
        img_height, img_width = img.shape[:2]
        img_channels = 1

    clf = joblib.load(learning_model_path)
    
   
    """ convert image """
    blur = cv2.GaussianBlur(img,(3,3),0)
    thresh_ = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,3)
    thresh = remove_noise(thresh_)
       
    kernel = np.ones((3,3),np.uint8)
    thresh_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh_closed = cv2.morphologyEx(thresh_opened, cv2.MORPH_CLOSE, kernel)
        
#    pl.gray()
#    pl.matshow(img)
#    pl.matshow(thresh_)
#    pl.matshow(thresh)
#    pl.matshow(thresh_opened)
#    pl.matshow(thresh_closed)
          
#    thresh = remove_noise(thresh_)
#    pl.matshow(thresh)
    
    thresh_final = thresh
    
    """ 反転対応  輝度の平均値から判断　"""
    ave1 = cv2.reduce(thresh_final,dim=0,rtype=1)
    ave2 = cv2.reduce(ave1,dim=1,rtype=1)
    if ave2 < 127:
        thresh_final = 255 - thresh_final 
    
#    pl.matshow(thresh_final)         #最終的な2値化画像を確認
    
    
    """ 文字領域の4端点を検出 """
    """ エッジ検出 """
    thresh_canny = cv2.Canny(thresh_final,100, 200)
#    pl.matshow(thresh_canny)
    
    img_show = img.copy() #表示用の画像
    cnts = cv2.findContours(thresh_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]  # 抽出した輪郭に近似するコンターを探す。
    cnts.sort(key=cv2.contourArea,reverse=True)  # 面積が大きい順に並べ替える。
    cnts.pop(0)
    #cnts.sort()
    
    warp = None
    point_min_pos = point_max_pos = (img_height + img_width) / 2
    point_min_neg = point_max_neg = (img_width - img_height) / 2
    for i, c in enumerate(cnts):
        if len(c) > img_height/10:
            arclen = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.001*arclen, True)
            cv2.drawContours(img_show, [approx], -1, (0, 0, 255), 2)
            
            for j in range(0,len(c)):
                #print(j," ,approx[j][0][0] , approx[j][0][1] = ",approx[j][0][0] , approx[j][0][1])
                
                point_pos = c[j][0][0] + c[j][0][1]
                point_neg = c[j][0][0] - c[j][0][1]
                
                if point_pos < point_min_pos:
                    point1 = [c[j][0][0] , c[j][0][1]]
                    point_min_pos = point_pos
                if point_pos > point_max_pos:
                    point3 = [c[j][0][0] , c[j][0][1]]
                    point_max_pos = point_pos
                if point_neg < point_min_neg:
                    point2 = [c[j][0][0] , c[j][0][1]]
                    point_min_neg = point_neg
                if point_neg > point_max_neg:
                    point4 = [c[j][0][0] , c[j][0][1]]
                    point_max_neg = point_neg
                
            if warp == None:
                warp = approx.copy()  # 一番面積の大きな四角形をwarpに保存。
   
    string_width_top = (point4[0] - point1[0])
    string_width_bottom = (point3[0] - point2[0])
    string_height_left = (point2[1] - point1[1])
    string_height_right = (point3[1] - point4[1])
    
    point1[0] = max(0, point1[0] -string_width_top/20)               
    point1[1] = max(0, point1[1]-string_height_left/2)    
    point2[0] = max(0, point2[0] -string_width_bottom/20)               
    point2[1] = min(img_height, point2[1]+string_height_left/2)
    point3[0] = min(img_width, point3[0]+string_width_bottom/20)               
    point3[1] = min(img_height, point3[1]+string_height_right/2)
    point4[0] = min(img_width, point4[0]+string_width_top/20)               
    point4[1] = max(0, point4[1]-string_height_right/2)
    
    
#    point1 = [10,10]
#    point2 = [5,70]
#    point3 = [1540,157]
#    point4 = [1540,80]
    points = (point1,point4,point3,point2)    
    
    #img_show = imresize(img_show,(1500/img_width))
    #cv2.imshow('edge', img_show)         #変形用の枠を確認
    
    warped = transform_by4(img, points)
    #pl.matshow(warped)

    
    thresh = transform_by4(thresh_final, points) 
    #thresh = thresh_final
#    pl.matshow(thresh)
    
    if len(thresh.shape) == 3:
        thresh_height, thresh_width, thresh_channels = warped.shape[:3]
    else:
        thresh_height, thresh_width = warped.shape[:2]
        thresh_channels = 1
    

#    cv2.line(img_show, (int(point1[0]),int(point1[1])), (int(point2[0]),int(point2[1])), (255,0,0), 1, 8, 0)
#    cv2.line(img_show, (int(point2[0]),int(point2[1])), (int(point3[0]),int(point3[1])), (255,0,0), 1, 8, 0)
#    cv2.line(img_show, (int(point3[0]),int(point3[1])), (int(point4[0]),int(point4[1])), (255,0,0), 1, 8, 0)
#    cv2.line(img_show, (int(point4[0]),int(point4[1])), (int(point1[0]),int(point1[1])), (255,0,0), 1, 8, 0)
#    pl.matshow(img_show)
#    img_show = imresize(warped,(1000/img_width))
#    cv2.imshow('warp', img_show)         #変形後の画像を確認

    
    
    """ 上下の幅を調節 """
    # 横に一列すべて255となる白線を検出
    # 左が白線なのに白線でなくなる座標、左が白線でないのに白線になる座標をそれぞれ(list)scan_start, (list)scan_endに記録する
    scanx_start = []
    scanx_end = []
    max_of_bright = thresh_width*255
    sum_of_bright = []
    sum_of_bright.append(max_of_bright)
    
    for scan in range(1,thresh_height):
        a = 0
        for scan_x in range(0,thresh_width):
            a += thresh[scan, scan_x]
        sum_of_bright.append(a)
        
        noise = int(max(1,thresh_width/50))
        if sum_of_bright[scan] <= (max_of_bright-(255*noise)) and sum_of_bright[scan-1] > (max_of_bright-(255*noise)):
            scanx_start.append(scan)
        elif sum_of_bright[scan] > (max_of_bright-(255*noise)) and sum_of_bright[scan-1] <= (max_of_bright-(255*noise)):
            scanx_end.append(scan)
    if (scanx_end == []) or (len(scanx_start)>len(scanx_end)):
        scanx_end.append(thresh_height)
    
    
    predict_string = []     #予測された文字列（URL)
    predict_result = []     #副候補も含めた結果
    
    for detect_x in range(0,len(scanx_start)):  
        if int(scanx_end[detect_x]-scanx_start[detect_x]) < max(10,(thresh_height/50)) :
            continue
             
        thresh_resize = thresh[max(scanx_start[detect_x]-(int)(scanx_end[detect_x]-scanx_start[detect_x])/7, 0): min(scanx_end[detect_x]+(int)(scanx_end[detect_x]-scanx_start[detect_x])/7,thresh_height) , : ]  
#        pl.matshow(thresh_resize)
        
        if len(thresh_resize.shape) == 3:
            thresh_resize_height, thresh_resize_width, thresh_resize_channels = thresh_resize.shape[:3]
        else:
            thresh_resize_height, thresh_resize_width = thresh_resize.shape[:2]
            thresh_resize_channels = 1
        
        """ scanning """
        # 縦に一列、すべての画素値が255(=白線)になるところを検出する
        # 左が白線なのに白線でなくなる座標、左が白線でないのに白線になる座標をそれぞれ(list)scan_start, (list)scan_endに記録する
        scan_start = []
        scan_end =[]
        
        max_of_bright = thresh_resize_height*255
        sum_of_bright = []
        sum_of_bright.append(max_of_bright)
        
        for scan in range(1,thresh_resize_width):
            a = 0
            for scan_y in range(0,thresh_resize_height):
                a += thresh_resize[scan_y, scan]
            sum_of_bright.append(a)
            
            noise = int(max(1,thresh_height/50))
            if sum_of_bright[scan] <= (max_of_bright-(255*noise)) and sum_of_bright[scan-1] > (max_of_bright-(255*noise)):
                scan_start.append(scan)
            elif sum_of_bright[scan] > (max_of_bright-(255*noise)) and sum_of_bright[scan-1] <= (max_of_bright-(255*noise)):
                scan_end.append(scan)
        if (scan_end == []) or (len(scan_start)>len(scan_end)):
            scan_end.append(thresh_height)               
        #print (scan_start) 
        #print (scan_end)
                       
        """複数の文字がつながってしまうときの対処"""
        #横長の文字は基本的に存在しないので、横長の領域はimg_height程度の大きさの領域に区切る
                
        ave_letter_width=0        
        for i in range(0,min(len(scan_start), len(scan_end))):
            ave_letter_width+=scan_end[i]-scan_start[i]
            
        if len(scan_start) !=0 :    ave_letter_width/=min(len(scan_start), len(scan_end))    
        
        
        j=0
        for i in range(0,min(len(scan_start), len(scan_end))-1):      
            """ノイズを文字としてしまったときの対処"""
            #あまりにも横幅が狭い場合削除        
            if (scan_end[j]-scan_start[j]) < max(10,ave_letter_width/10):
                del scan_start[j]
                del scan_end[j]
                j-=1
            j+=1    
        
        ave_letter_width=0        
        for i in range(0,min(len(scan_start), len(scan_end))):
            ave_letter_width+=scan_end[i]-scan_start[i]
            
        if len(scan_start) !=0 :    ave_letter_width/=min(len(scan_start), len(scan_end))    
        
        for i in range(0,min(len(scan_start), len(scan_end))):
            if (scan_end[i]-scan_start[i]) > ave_letter_width*2 :
                print("yokonaga!",detect_x,i)
                devide_number = math.floor((scan_end[i]-scan_start[i]) / ave_letter_width)      #分割数
                devide_scale = int((scan_end[i]-scan_start[i]) / devide_number)
                
                for j in range(1,devide_number):
                    scan_end.insert( (i+j-1) ,  (scan_start[i]+devide_scale*(j)) )
                    
                for k in range(1,devide_number):
                    scan_start.insert( (i+k) , (scan_start[i]+devide_scale*k) )     
                
            
        """ detected """
        detected_images = []
        
        
        for detect in range(0,len(scan_start)):
            
            # scan_start[detect]とscan_end[detect]の間の領域が、detect番目に発見された文字領域である
            cv2.rectangle(warped, (scan_start[detect], max(int(scanx_start[detect_x]-int(scanx_end[detect_x]-scanx_start[detect_x])/5), 0) ),(scan_end[detect] ,min(int(scanx_end[detect_x]+(int)(scanx_end[detect_x]-scanx_start[detect_x])/5),thresh_height)    ),(0,0,255),2)
            
            detected_image = thresh_resize[0:thresh_resize_height,scan_start[detect]:scan_end[detect]]  
            if len(detected_image.shape) == 3:
                detect_height, detect_width, detect_channels = detected_image.shape[:3]
            else:
                detect_height, detect_width = detected_image.shape[:2]
                detect_channels = 1    
            #pl.matshow(detected_images[detect])
        
            detected_images.append(detected_image)
            
            if detect_width <= detect_height : 
                resize = np.ones([detect_height,detect_height])
                resize = resize*255
                
                resize[0:detect_height , (detect_height/2-detect_width/2):(detect_height/2+detect_width/2)] = detected_image
               
                detected_resize = imresize(resize, (20,20) )
            else:
                resize = np.ones([detect_width,detect_width])
                resize = resize*255
                
                resize[(detect_width/2-detect_height/2):(detect_width/2+detect_height/2) , 0:detect_width] = detected_image
               
                detected_resize = imresize(resize, (20,20) )
                
                
                
        #        resize = imresize(detected_image,(20.0/detect_width))
        #        if len(resize.shape) == 3:
        #            resize_height, resize_width, resize_channels = resize.shape[:3]
        #        else:
        #            resize_height, resize_width = resize.shape[:2]
        #            resize_channels = 1        
        #       
        #        detected_resize =  np.ones([20,20],dtype=np.uint8 )
        #        detected_resize = detected_resize*255
        #        detected_resize[((20/2)-resize_height/2):((20/2)+resize_height/2),0:20] = resize
            
#            pl.matshow(detected_resize)
            
            a = np.reshape(detected_resize,(detected_resize.shape[0]*detected_resize.shape[1]))           
            predict = clf.predict(a)[0]
            classes = clf.classes_
            pred_proba = clf.predict_proba(a)
            result_matrix=np.vstack((classes,pred_proba))
            result_matrix = np.delete(result_matrix,np.where(result_matrix[1,:]<0.1),1)
            #result_matrix = result_matrix[:2,:]
            predict = []
            # print("文字番号 : ",detect)
            for i in range(len(result_matrix[1,:])):
                index_max = np.argmax(result_matrix[1,:])
                if i == 0:  predict_string.append(result_matrix[0,index_max])  
                #print(result_matrix[:2,index_max])
                predict.append(result_matrix[:2,index_max])
                result_matrix = np.delete(result_matrix,index_max,1)
            #print("")
            
            predict_result.append(predict)
        
        #print(predict_string)
        predict_result.append([])
        predict_string.append([])
        
    PREDICT_RESULT=pd.DataFrame(predict_result)
    PREDICT_RESULT.to_csv(csv_result_path,index=0,header=None)    
    
    PREDICT_STRING=pd.DataFrame(predict_string)
    PREDICT_STRING.to_csv(csv_string_path,index=0,header=None,line_terminator='\t',sep='\t')  
    
    img_show = imresize(warped,(1500/img_width))
#    cv2.imshow('norm', img_show)
    cv2.waitKey(0) # 1msec待つ
    cv2.destroyAllWindows()
    
