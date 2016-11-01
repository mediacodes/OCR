# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
#from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

########## Loading ###################################
#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
import pandas as p
import time as time

print("Now Loading Data")
start_time = time.clock()

csv_test_path = 'C:\\Users\\mech-user\\Documents\\AlphabetsRecognition\\alldata_resized_third(20).csv'
X = np.array(p.read_csv(filepath_or_buffer=csv_test_path, header=None, sep=',', index_col=0))[:, :]
y = np.ravel(np.array(p.read_csv(filepath_or_buffer=csv_test_path, header=None, sep=',', usecols=[0]))[:, :])

#データ変形
X = X/255
data_num = X.shape[0]
X.resize(81000, 1, 20, 20)
print("shape:", X.shape)
#for i in range(0,data_num):
#    matrix = X[i] 
#    matrix.resize(1, 20, 20)
#print("second:",X[0][0])

##print("second:/n",X[0])
#X.transpose(3, 0, 1, 2)
#print("third:/n",X[0])
#for j in X:
#    j.reshape(1, 20, 20)
alphabets=['zero','one','two','three','four','five','six','seven','eight','nine','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','apostrophe','and','asterisk','at','cl_parenthesis','colon','comma','dollar','dot','equal','exclamation','hyphen','op_parenthesis','percent','plus','question','semi-colon','slash','underbar']
j = 0
for i in y: 
    y[j] = int(alphabets.index(i))
    j += 1
y = y.astype(dtype=int) 
    
    
from sklearn.cross_validation import train_test_split
x_train,x_test,t_train,t_test = train_test_split(X,y,test_size = 0.2, random_state=42)
#train_data,test_data,train_label,test_label = train_test_split(X,y,test_size = 0.2, random_state=42)

end_time = time.clock()
print("Loading Complete \nTime =", end_time - start_time)


# 処理に時間のかかる場合はデータを削減 
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

########## Learning ###################################
print("Now Learning")
start_time = time.clock()
max_epochs = 50 #元は20

network = SimpleConvNet(input_dim=(1,20,20), 
                        conv_param = {'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=81, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

end_time = time.clock()
print("Learning Complete \nTime =", end_time - start_time)

# パラメータの保存
network.save_params("params_cnn2.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
