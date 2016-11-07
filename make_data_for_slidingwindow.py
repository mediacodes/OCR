# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

chars_path = 'C:/Users/Nantan/Documents/mediaseek/OCR/data/characters_20by20.csv'
space_data_path = 'C:/Users/Nantan/Documents/mediaseek/OCR/data/space_data_20by20.csv'

allchars = pd.read_csv(chars_path, header=None, index_col=0)
allchars = allchars.as_matrix()
space_data =[]
#print(np.size(allchars, 0))
for i in range(np.size(allchars, 0)):
    for j in range(i, np.size(allchars, 0)):
        _img1 = allchars[i, :].reshape(20, 20)
        _img2 = allchars[j, :].reshape(20, 20)
        _space_data1 = ((np.hstack((_img1, _img2)))[:, 10:30]).flatten()
        _space_data2 = ((np.hstack((_img2, _img1)))[:, 10:30]).flatten()
        space_data.append(_space_data1)
        space_data.append(_space_data2)
        break
    break

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