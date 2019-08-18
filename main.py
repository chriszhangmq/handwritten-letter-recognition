#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Chris'

import src
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split   # 分割数据集
from sklearn.preprocessing import LabelBinarizer       # 标签二值化

'''*************************图像处理**********************************'''
print("============processing image============")
src.process_image.image_to_vector()
print("================end===============")

'''*************************导入数据**********************************'''
print("============load data============")
load_data = src.file_operation.load_data_num('./data.txt')
data = load_data[:, 1:]
label = load_data[:, 0]
print("The dimension of data= %s  \nThe dimension of label= %s" %(data.shape, label.shape))

'''*************************数据处理**********************************'''
print("============processing data============")
# 分割数据：分为train：（训练数据集），val：（交叉验证集）,test：（测试数据集）
X_data, X_test, y_data, y_test = train_test_split(data, label, test_size=0.2)   # 测试集占总样本百分比
X_train = X_data[0:40000]   # X_data是全部数据集的80%
y_train = y_data[0:40000]
X_val = X_data[40000:-1]
y_val = y_data[40000:-1]
print("The dimension of train data= %s\n"
      "The dimension of cross validation data= %s\n"
      "The dimension of test data= %s\n" %(X_train.shape, X_val.shape, X_test.shape))
# 进行标签二值化,共26列，对应26个字母，1-->[0,1,0,0,0,0....,0,0,0,0]   5=[0,0,0,0,0,1,0,0,....0,0]
y_train_label = LabelBinarizer().fit_transform(y_train)

'''*************************创建神经网络**********************************'''
print("=========Create BP neural network==========")
classify = src.NetFunction.NeuralNetwork(data.shape[1], 300, 26)

print('=======training========')
loss_list, accuracy_train, accuracy_val = classify.train(X_train, y_train_label, y_train, X_val, y_val, learn_rate=0.1,num_iters=10000)
print('==========end==========')

'''*************************可视化数据**********************************'''
plt.subplot(211)   #绘制两个图形，（2，1），当前窗口编号为1，绘图
plt.plot(loss_list, label='train loss')
plt.title('train loss')
plt.xlabel('iters')
plt.ylabel('loss')
plt.legend(loc='lower right')

plt.subplot(212)   #绘制两个图形，（2，1），当前窗口编号为2，绘图
plt.plot(accuracy_train, label='train_acc', color='red')
plt.plot(accuracy_val, label='val_acc', color='black')
plt.xlabel('iters')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()

'''*************************预测数据*********************************'''
# 预测精度
y_pred = classify.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("the accuracy is : ", accuracy)

# 随机挑选样本，检测参数
print("the random sample selection,the sample serial number is：")
m, n = data.shape
example_size = 30
example_index = np.random.choice(m, example_size)
print(example_index)
for i, idx in enumerate(example_index):
    print("%dth example is number %d,we predict it as %d" \
    % (i, label[idx], classify.predict(data[idx, :].reshape(1, -1))))
