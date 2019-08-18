#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Chris'

import numpy as np
import matplotlib.pyplot as plt

letter = ['A','B','C','D','E','F','G','H','I','J',
          'K','L','M','N','O','P','Q','R','S','T',
          'U','V','W','X','Y','Z']

def load_data(filename):
    """
    打开文件函数
    :param filename: 输入文件路径
    :return: 返回array，对应的数据矩阵
    """
    data = []
    file = open(filename)
    for line in file.readlines():
        lineArr = line.strip().split(',')
        col_num = len(lineArr)
        temp = []
        for i in range(col_num):
            if lineArr[i].isdigit() == True:
                temp.append(float(lineArr[i]))
            else:
                temp.append(letter.index(lineArr[i]))
        data.append(temp)
    return np.array(data)

def load_data_num(filename):
    """
    打开文件函数
    :param filename: 输入文件路径
    :return: 返回array，对应的数据矩阵
    """
    data = []
    file = open(filename)
    for line in file.readlines():
        lineArr = line.strip().split(',')
        col_num = len(lineArr)
        temp = []
        for i in range(col_num):
            temp.append(float(lineArr[i]))
        data.append(temp)
    return np.array(data)

def display_data(x,y):
    """
    数据显示函数
    :param x: 横坐标数据
    :param y: 纵坐标数据
    """
    label_0 = np.where(y.ravel() == 0)
    plt.scatter(x[label_0, 0], x[label_0 , 1], marker='x', color='r', label='Not admitted' )
    label_1 = np.where(y.ravel() == 1)
    plt.scatter(x[label_1, 0], x[label_1 , 1], marker='o', color='b', label='Admitted' )
    plt.xlabel("Exam_1 score")
    plt.ylabel("Exam_2 score")
    plt.legend(loc='upper left')  # 显示图例位置
    plt.show()


#测试部分
def test():
    data = load_data('../data/ex2data1.txt')
    print(data.shape)
    print(data[:5])
    display_data(data[:,:-1],data[:,-1:])
    return data


if __name__ == '__main__':
    test()