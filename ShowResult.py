import pickle
import torch
import torchvision
import torch.onnx
import matplotlib.pyplot as plt
import os

import time
import numpy as np

fileAndPath = 'fruits-360_channel-3_test_acc-82.68_ThreeLayerConvNetfinal_pack_20201107_13-29-39.pickle'


result = torch.load(fileAndPath)

cnt = 0
for item in result:
    print('The data ', cnt, ' is : ', item)
    cnt += 1


print(result['loss record'])

fig, axs = plt.subplots(nrows=1, ncols=1)
axs.plot(np.arange(len(result['loss record'])), result['loss record'])
plt.show()