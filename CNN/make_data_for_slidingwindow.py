# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.misc import imread,imresize


chars_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\CSV\\CSV_20,clean\\alldata\\alldata_20clean.csv'
space_data_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\CSV\\CSV_slidingwindow\\space_data_20by20noise4.csv'
letter_data_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\CSV\\CSV_slidingwindow\\letter_data_20by20_2.csv'


allchars = pd.read_csv(chars_path, header=None, index_col=0)
allchars = allchars.as_matrix()
space_data =[]
letter_data = []
#print(np.size(allchars, 0))

""" 0ならspace作り 1ならletter作り """
space_or_letter = 1

if(space_or_letter == 1):
    counter = 0
    for i in range(0, np.size(allchars, 0)):
        img1 = np.zeros((20,20))
        _img1 = allchars[i, :].reshape(20, 20) 
        for x in range(0,20):
            for y in range(0,20):
                #0~255の範囲から出ないように画素値を±10する 
                b = min(255,max(0,(_img1[x][y] + random.randint(-10,10))))
                img1[x][y] = b
 
  #      img1_resize1 = imresize(img1[:,4:16], (20, 20)).flatten()

        
        for yokohaba in range(3,7):
            
            for size in range(0,4):
                resize = imresize(img1[:,yokohaba:(20-yokohaba)], (20, 20))
                 #imresize
                detected_resize = imresize(resize, ((20-size*2),20 ) )
                
                #領域の初期化
                img1_resize1 = np.ones([20,20],dtype=np.uint8)
                img1_resize1 = img1_resize1*255                     
                
                #元画像をずらして配置
                img1_resize1[max(0,size):min(20,20-size) , :] = detected_resize[max(0,-size):min(20-size*2,20-size) , :]
         
                counter += 1
                if ((counter % 500) ==0):
                    plt.matshow(img1_resize1)
                letter_data.append(img1_resize1.flatten())
        
        
#        img1_resize2 = imresize(img1[:,3:17], (20, 20)).flatten()
#        counter += 1
#        if ((counter % 200) ==0):
#            plt.matshow(img1_resize2.reshape(20,20))
#        letter_data.append(img1_resize2)
#        
#        img1_resize3 = imresize(img1[:,5:15], (20, 20)).flatten()
#        counter += 1
#        if ((counter % 200) ==0):
#            plt.matshow(img1_resize3.reshape(20,20))
#        letter_data.append(img1_resize3)
#        
#        img1_resize4 = imresize(img1[:,6:14], (20, 20)).flatten()
#        counter += 1
#        if ((counter % 200) ==0):
#            plt.matshow(img1_resize4.reshape(20,20))
#        letter_data.append(img1_resize4)


    print("counter=",counter)
    letter_data = pd.DataFrame(letter_data)
    letter_data.to_csv(letter_data_path, header=False, index=False)               
    
else:
    
    counter = 0
    for i in range(0, np.size(allchars, 0), 3):
        for j in range(i, np.size(allchars, 0), 3):
            img1 = np.zeros((20,20))
            img2 = np.zeros((20,20))
            _img1 = allchars[i, :].reshape(20, 20)        
            _img2 = allchars[j, :].reshape(20, 20)
            for x in range(0,20):
                for y in range(0,20):
                    #0~255の範囲から出ないように画素値を±10する 
                    b = min(255,max(0,(_img1[x][y] + random.randint(-10,10))))
                    img1[x][y] = b
                    b = min(255,max(0,(_img2[x][y] + random.randint(-10,10))))
                    img2[x][y] = b
                    
            space_data1 = ((np.hstack((img1, img2)))[:, 10:30]).flatten()
            space_data2 = ((np.hstack((img2, img1)))[:, 10:30]).flatten()
    #        space_data.append(space_data1)
    #        space_data.append(space_data2)
            counter += 1
            if ((counter % 5000) ==0):
                plt.matshow(space_data1.reshape(20,20))
            
            img1_resize1 = img1[:,:17]
            img2_resize1 = img2[:,3:]
            space_data3 = ((np.hstack((img1_resize1, img2_resize1)))[:, 7:27]).flatten()
            space_data.append(space_data3)
            if ((counter % 5000) ==0):
                plt.matshow(space_data3.reshape(20,20))
    
            img1_resize2 = img1[:,:15]
            img2_resize2 = img2[:,5:]
            space_data4 = ((np.hstack((img1_resize2, img2_resize2)))[:, 5:25]).flatten()
            space_data.append(space_data4)
            if ((counter % 5000) ==0):
                plt.matshow(space_data4.reshape(20,20))
                
            img1_resize3 = img1[:,:18]
            img2_resize3 = img2[:,2:]
            space_data4 = ((np.hstack((img1_resize3, img2_resize3)))[:, 8:28]).flatten()
            space_data.append(space_data4)
            if ((counter % 5000) ==0):
                plt.matshow(space_data4.reshape(20,20))
    
    print("counter=",counter)
    space_data = pd.DataFrame(space_data)
    space_data.to_csv(space_data_path, header=False, index=False)
    
    """
    img1 = img.iloc[0, :]
    img2 = img.iloc[1, :]
    img1 = img1.as_matrix().reshape(20, 20)
    img2 = img2.as_matrix().reshape(20, 20)
    img3 = np.hstack((img1, img2))
    newimg = img3[:, 10:31]
    plt.imshow(img3, cmap='gray', interpolation='none')
    plt.show()
    plt.imshow(newimg, cmap='gray', interpolation='none')
    plt.show()
    """