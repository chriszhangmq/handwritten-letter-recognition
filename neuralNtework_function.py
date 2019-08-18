#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Chris'

import numpy as np

class NeuralNetwork(object):
    #初始化:该网络只有3层
    def __init__(self, input_size, hidden_size, output_size):
        self.theta1 = 0.01 * np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.theta2 = 0.01 * np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def sigmoid(self, x):
        """
        激活函数sigmoid
        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        """
        sigmoid函数求导
        :param x:
        :return:
        """
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    #代价函数
    def cost(self, x , y, reg = 250):
        """
        代价函数
        :param x: 训练数据
        :param y: 标签
        :param reg: 正则化系数,1/reg
        """
        num_train, num_feature = x.shape
        #前向传播
        a1 = x
        a2 = self.sigmoid(a1.dot(self.theta1) + self.b1)
        a3 = self.sigmoid(a2.dot(self.theta2) + self.b2)

        #计算代价函数:添加正则化
        cost = -1 * np.sum((y * np.log(a3) + (1 - y) * np.log(1 - a3))) / num_train
        cost += (np.sum(self.theta1 * self.theta1) + np.sum(self.theta2 * self.theta2)) / (2 * num_train * reg)

        #反向传播
        error3 = a3 - y  # N*C
        d_theta2 = a2.T.dot(error3) + self.theta2 / reg  # (H*N)*(N*C)=H*C  核心公式1，求theta2的梯度相当于dJ / d(theta2)=a2.T.dot(error3)+lamba.theta2（算上正则化）
        db2 = np.sum(error3, axis=0)

        error2 = error3.dot(self.theta2.T) * self.d_sigmoid(a2)  # N*H    核心公式2 求误差
        d_theta1 = a1.T.dot(error2) + self.theta1 / reg  # (D*N)*(N*H) =D*H
        db1 = np.sum(error2, axis=0)

        d_theta1 /= num_train
        d_theta2 /= num_train
        db1 /= num_train
        db2 /= num_train
        return cost, d_theta1, d_theta2, db1, db2

    def train(self, x, y, y_train, x_val, y_val, learn_rate = 0.1, num_iters = 5000):
        """
        训练函数
        :param x:
        :param y:
        :param y_train:
        :param x_val:
        :param y_val:
        :param learn_rate:
        :param num_iters:
        :return:
        """
        batch_size = 200       # 批尺寸，批训练，不是一次性将全部数据进行训练
        num_train = x.shape[0]  # 训练样本
        loss_list = []
        accuracy_train = []
        accuracy_val = []
        for i in range(num_iters):  # 训练的次数10000次
            # 每次随机的选取小样本batch set去训练
            batch_index = np.random.choice(num_train, batch_size, replace=True)  # 生成的随机数中可以有重复的数值
            x_batch = x[batch_index]  # 150*64
            y_batch = y[batch_index]
            y_train_batch = y_train[batch_index]

            loss, d_theta1, d_theta2, db1, db2 = self.cost(x_batch, y_batch)
            loss_list.append(loss)
            # update the weight
            self.theta1 += -learn_rate * d_theta1
            self.theta2 += -learn_rate * d_theta2
            self.b1 += -learn_rate * db1
            self.b2 += -learn_rate * db2

            if i % 500 == 0:  # 每500次，进行一次统计
                print("The number of trianing: %dth, loss=%f" % (i, loss))
                # 记录训练精度、交叉验证精度
                train_acc = np.mean(y_train_batch == self.predict(x_batch))
                val_acc = np.mean(y_val == self.predict(x_val))
                accuracy_train.append(train_acc)
                accuracy_val.append(val_acc)

        return loss_list, accuracy_train, accuracy_val

    def predict(self, x_test):
        """
        预测函数
        :param x_test:
        :return:
        """
        a1 = x_test
        a2 = self.sigmoid(a1.dot(self.theta1) + self.b1)
        a3 = self.sigmoid(a2.dot(self.theta2) + self.b2)
        y_pre = np.argmax(a3, axis=1)   #axis=1,代表行，即：返回行最大的标号
        return y_pre



