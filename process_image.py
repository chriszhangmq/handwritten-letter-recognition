#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Chris'

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import scipy.misc as smi


def binarization(imgs):
    """
    二值化函数
    :param imgs:
    :return:binarization matrix
    """
    img_bin = []
    for i in range(len(imgs)):
        img = imgs[i]
        mat_mean = np.mean(img)
        img[img <= mat_mean] = 0
        img[img >= mat_mean] = 1
        img_bin.append(img)

    return np.array(img_bin)

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
            temp.append(float(lineArr[i]))
        data.append(temp)
    return np.array(data)

def my_imresize( imgs, condense_rows, condense_cols):
    """
    图片尺寸修改
    :param imgs:
    :param condense_rows:
    :param condense_cols:
    :return:
    """
    imgs_conds = []
    img = imgs
    img_temp = smi.imresize(img, (condense_rows, condense_cols),mode="L")
    imgs_conds.append(img_temp)

    return np.array(imgs_conds)

def get_imlist(path):
    """
    此函数读取特定文件夹下的jpg格式图像地址信息，存储在列表中
    :param path:
    :return:
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]

def get_dirlist(dir):
    """
    得到路径下文件夹的数量
    :param dir:
    :return:
    """
    path = os.getcwd()  # 获取当前路径
    print(path)
    count_dir = 0
    for root, dirs, files in os.walk(path + dir ):  # 遍历统计
        for each in dirs:
            count_dir += 1  # 统计文件夹下文件个数
    return count_dir

def image_to_vector():
    """
    将图片转换为向量（矩阵）
    :return:
    """
    # 文件数目
    num_dir_captials = get_dirlist('./img/new_letter/') #52
    #每个文件夹内的图片数目
    num_per_dir_captials = 1016
    #总的图片数目: num_per_dir_captials * num_dir_captials
    num_img = 0
    #字母序号0-25
    num_letter = 0
    # 建立d*（1,256*256*3）的随机矩阵
    data = np.empty((1, 256 * 256 * 3))
    #最终存放的数据
    svae_data = np.zeros((num_per_dir_captials * num_dir_captials, (16*16+1)))

    for num_dir in range(num_dir_captials):
        path = os.getcwd()  # 获取当前路径
        image = get_imlist((path + ('/Img/new_letter/Sample0%d' % (num_dir + 11))))
        print(path + ('/Img/new_letter/Sample0%d' % (num_dir + 11)))

        for num_per_img in range(num_per_dir_captials):
            img = Image.open(image[num_per_img]).convert('L')  # 打开图像
            img_ndarray = np.asarray(img, dtype='float64')   # 将图像转化为数组并将像素转化到0-1之间
            img_ndarray = 255-img_ndarray
            img_ndarray = binarization(img_ndarray) #二值化

            data = np.ndarray.flatten(img_ndarray)  # 将图像的矩阵形式转化为一维数组保存到data中

            svae_data[num_img, 0] = num_letter #对应的字母标志A-Z :0-25   a-z : 26-51
            svae_data[num_img, 1:] = np.array(data).reshape(1, -1)  # 将一维数组转化为矩阵
            num_img += 1
        num_letter += 1
        if num_letter >= 26:
            num_letter =0
    savetxt('data.txt', svae_data, fmt="%d", delimiter=',')  # 进行存储保存为两位小数，数据以","为间隔
    return 0


