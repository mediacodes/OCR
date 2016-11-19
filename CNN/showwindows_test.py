# -*- coding: utf-8 -*-

from sklearn.externals import joblib
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imresize
import numpy as np
import pandas as pd

img_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\slidingwindow\\slidwindows\\window'
learning_model_path = 'sliding_windows1(20by6).pkl'

"""
img_path = 'C:/Users/Nantan/OneDrive/Documents/mediaseek/OCR/slidingwindow/slidwindows/window'
learning_model_path = 'C:/Users/Nantan/OneDrive/Documents/mediaseek/OCR/slidingwindow/Classifier/SVM/LearningModel/default/default'
"""

network = SimpleConvNet(input_dim=(1,20,20), 
                        conv_param = {'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=81, weight_init_std=0.01)
                        
# パラメータの読み込み
network.load_params(learning_model_path)


for i in range(54): 
    _img_path = img_path + str(i) + '.jpg'
    img = cv2.imread(_img_path, 0)
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.show()
    reimg = imresize(img, (20, 20)).flatten() / 255.0
    
    reimg = np.resize(reimg,(20,20))
    
    data = np.zeros((1,1,20,20))
    data[0][0] = reimg
    
    for layer in network.layers.values():
        data = layer.forward(data)
    
    data = softmax(data)
    result = data[0]

    predict_pd = pd.DataFrame(result)
    predict_pd.index = ['letter','space']
    predict = predict_pd.sort_values(by=0, ascending=False)

    print(predict[0])
