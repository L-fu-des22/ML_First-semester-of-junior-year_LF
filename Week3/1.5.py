```python
from PIL import Image
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
# import tensorflow as tf
from sklearn.metrics import classification_report  # 这个包是评价报告
import scipy.optimize as opt

# this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(strv):
    directory_name_1 = 'E:\Junior first semester\\374-Machine Learning System Design\week3\Homework3\Homework3\data\Digits\\'+strv+'\\1'
    directory_name_2 = 'E:\Junior first semester\\374-Machine Learning System Design\week3\Homework3\Homework3\data\Digits\\'+strv+'\\2'
    array_of_img = []
    for filename in os.listdir(directory_name_1):
        img = cv2.imread(directory_name_1 + "\\" + filename)
        inf = np.array(img).flatten()
        inf = np.concatenate((inf, [1]))
        array_of_img.append(inf)
    for filename in os.listdir(directory_name_2):
        img = cv2.imread(directory_name_2 + "\\" + filename)
        inf = np.array(img).flatten()
        inf = np.concatenate((inf, [0]))
        array_of_img.append(inf)
    x = np.array(array_of_img)
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((ones, x), axis=1)
    return x

def get_x(df):#读取特征
    data = df[:, :-2]
    return data

def get_y(df):
    y = df[:, -1]
    return y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)

def gradient(theta, X, y):  # just 1 batch gradient
    return (1 / len(X)) * (X.T @ (sigmoid(X @ theta) - y))

def regularized_cost(theta, X, y, l=1):
    #     '''you don't penalize theta_0'''
    theta_j1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

    return cost(theta, X, y) + regularized_term  # 正则化代价函数

def regularized_gradient(theta, X, y, l=1):
    #     '''still, leave theta_0 alone'''
    # regularized gradient(正则化梯度)
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term

''
def feature_mapped_logistic_regression(l, data_train, data_test):
    X_train = get_x(data_train)
    y_train = get_y(data_train)
    print(X_train.shape)
    print(y_train.shape)
    theta = np.zeros(X_train.shape[1])
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X_train, y_train, l),
                       method='TNC',
                       jac=regularized_gradient)

    final_theta = res.x
    print('The solution of the optimization is ')
    print(final_theta)
    # 训练集预测模型报告
    y_pred = predict(X_train, final_theta)
    print('训练集预测报告：')
    print(classification_report(y_train, y_pred))

    # 读取测试集，并测试训练集生成的模型
    X_test = get_x(data_test)
    y_test = get_y(data_test)

    y_test_pred = predict(X_test, final_theta)
    print('测试集预测报告：')
    print(classification_report(y_test, y_test_pred))
    return final_theta

if __name__ == '__main__':
    train = 'train'
    test = 'test'

    data_train = read_directory(train)
    data_test = read_directory(test)

    l = 1 #惩罚系数
    feature_mapped_logistic_regression(l, data_train, data_test)
```