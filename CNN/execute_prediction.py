# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:33:39 2016

@author: mech-user
"""

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as pl
#from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
from common.functions import *
from operator import itemgetter
import pandas as pd

########## Loading ###################################
#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
import pandas as p
import time as time

print("Now Loading Data")
start_time = time.clock()

csv_test_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\alldata_resized_third(20).csv'
X = np.array(p.read_csv(filepath_or_buffer=csv_test_path, header=None, sep=',', index_col=0))[41400, :]
y = np.ravel(np.array(p.read_csv(filepath_or_buffer=csv_test_path, header=None, sep=',', usecols=[0]))[41400, :])

#データ変形
X = X/255
data_num = X.shape[0]
X.resize(1, 1, 20, 20)

alphabets=['zero','one','two','three','four','five','six','seven','eight','nine','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','apostrophe','and','asterisk','at','cl_parenthesis','colon','comma','dollar','dot','equal','exclamation','hyphen','op_parenthesis','percent','plus','question','semi-colon','slash','underbar']
j = 0
label_num = np.zeros(y.shape)
for i in y: 
    label_num[j] = int(alphabets.index(i))
    j += 1
label_num = label_num.astype(dtype=int) 
    
    
from sklearn.cross_validation import train_test_split
#x_train,x_test,t_train,t_test = train_test_split(X,y,test_size = 0.2, random_state=42)
#train_data,test_data,train_label,test_label = train_test_split(X,y,test_size = 0.2, random_state=42)

end_time = time.clock()
print("Loading Complete \nTime =", end_time - start_time)



########## Learning ###################################
print("Now Predicting")
start_time = time.clock()

network = SimpleConvNet(input_dim=(1,20,20), 
                        conv_param = {'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=81, weight_init_std=0.01)
                        
# パラメータの読み込み
network.load_params("params_cnn3.pkl")

data = X
for layer in network.layers.values():
    data = layer.forward(data)

data = softmax(data)
result = data[0]

predict_pd = pd.DataFrame(result)
predict_pd.index = alphabets
predict = predict_pd.sort_values(by=0, ascending=False)

#predict = network.print_prediction(X)
print("label = ", y)
pl.gray
pl.matshow(X[0][0])
print("predict = ")
print(predict)
end_time = time.clock()
print("Learning Complete \nTime =", end_time - start_time)

