#coding:utf-8
## Making CSV file from the images
import numpy as np
import time as time
from scipy.misc import imread,imresize
import csv


#http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k
#64 classes (0-9, A-Z, a-z)
#7705 characters obtained from natural images
#3410 hand drawn characters using a tablet PC
#62992 synthesised characters from computer fonts   we use this data

data_path = 'Chars74K_EnglishFnt/data_img/Sample'
csv_test_path = 'Chars74K_EnglishFnt/data_csv/csv_test_imresize(imresize0.5,0.2)/'
csv_train_path = 'Chars74K_EnglishFnt/data_csv/csv_train_imresize(imresize0.5,0.2)/'

alphabets=['zero','one','two','three','four','five','six','seven','eight','nine','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


for j in range(0,62):
    if j > 8:
        data_path1 = data_path + '0' + str(j+1) + '/img0' + str(j+1) +'-'
        csv_test_path1 = csv_test_path + 'test0' + str(j+1) +'.csv'
        csv_train_path1 = csv_train_path +'train0' + str(j+1) +'.csv'
    else:
        data_path1 = data_path + '00' + str(j + 1) + '/img00' + str(j + 1) + '-'
        csv_test_path1 = csv_test_path + 'test00' + str(j + 1) + '.csv'
        csv_train_path1 = csv_train_path + 'train00' + str(j + 1) + '.csv'

    start_time = time.clock() 

    # Making CSVs
    print("Making CSV",j,"...\n")
    path = data_path1 + '00001' +'.png'
    img = imread(path)
    img=imresize(imresize(img,50),20)
    a = np.reshape(img,144)
    #print(a)
    #print(a.shape)
    csvfile =  [alphabets[j]]
    csvfile = np.r_[csvfile, a]

    with open( csv_test_path1, 'w') as f:
        writer = csv.writer(f, lineterminator='\n') 
        writer.writerow(csvfile)   
        #writer.writerow('\n')   

    for i in range(2,1000):
        if i<10:
            path = data_path1 + '0000' + str(i) +'.png'
        elif i<100:
            path = data_path1 + '000' + str(i) +'.png'
        elif i<1000:
            path = data_path1 + '00' + str(i) +'.png'
        else:
            path = data_path1 + '0' + str(i) +'.png'
   
        img = imread(path)
        img=imresize(imresize(img,50),20)
        a = np.reshape(img,144)
        #print(i,a)
        #print(a.shape)

        csvfile =  [alphabets[j]]
        csvfile = np.r_[csvfile,a] 

        if i<100:
            with open( csv_test_path1, 'a') as f:
                writer = csv.writer(f, lineterminator='\n') 
                writer.writerow(csvfile)   
                #writer.writerow('\n') 
        else:
            with open( csv_train_path1, 'a') as f:
                writer = csv.writer(f, lineterminator='\n') 
                writer.writerow(csvfile)   
                #writer.writerow('\n')   

    end_time =  time.clock()
    print("Making CSV Complete \nTime =",end_time-start_time)
    #Time =   13.19833485916505



