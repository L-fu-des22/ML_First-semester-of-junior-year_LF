
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
import matplotlib.pyplot as plt
# import tensorflow as tf
from sklearn.metrics import classification_report  # 这个包是评价报告
import scipy.optimize as opt

def get_X(df):  # 读取特征 , 返回data是个2维矩阵
    ones = pd.DataFrame({'ones': np.ones(len(df))})#ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    # data = data.drop([3,8,10,12, 13, 18, 21, 25],axis=1)
    # data = np.array(data.iloc[:, :-3])
    # # data = np.delete(data, [3,8,10,12, 13, 18, 21, 25], axis=1)
    # return data # 这个操作返回 ndarray,不是矩阵   最后一行是-3   2,3,4, 5,6,17
    # ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    X_data = pd.concat([ones, data.iloc[:, 1:-3]], axis=1) #1~27
    # # print(X_data)
    # # print(np.array(X_data).reshape(np.array(data).shape[0], 2))
    # # return np.array(X_data).reshape(np.array(data).shape[0], 2)  # 这个操作返回 ndarray,不是矩阵

    X_data = np.array(X_data)

    return X_data

def get_y(df):  # 读取标签, 返回y是一维矩阵
    return np.array(df.iloc[:, -1])  # df.iloc[:, -1]是指df的最后一列

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''

    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


def predict(x, theta):

    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)


def gradient(theta, X, y):  # just 1 batch gradient
    return (1 / len(X)) *( X.T @ (sigmoid(X @ theta) - y))


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


def feature_mapped_logistic_regression(l, data_train, data_test):
    X_train = get_X(data_train)
    y_train = get_y(data_train)
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

    y_test = get_y(data_test)
    X_test = get_X(data_test)

    y_test_pred = predict(X_test, final_theta)
    print('测试集预测报告：')
    print(classification_report(y_test, y_test_pred))
    return final_theta

data_train = pd.read_excel(
    r'E:\Junior first semester\374-Machine Learning System Design\week3\Homework3\Homework3\data\Disease\data_2.xlsx',
    sheet_name='train')

data_test = pd.read_excel(
    r'E:\Junior first semester\374-Machine Learning System Design\week3\Homework3\Homework3\data\Disease\data_2.xlsx',
    sheet_name='test')



l = 0.4 # lambda=惩罚系数
final_theta = feature_mapped_logistic_regression(l, data_train, data_test)



# sns.lmplot('TC', 'LDL_C', hue='CEREBRAL_APOPLEXTY', data=df, size=6, fit_reg=False, scatter_kws={"s": 100})
#
# plt.scatter(x, y, c='Red', s=1)
# plt.title('Decision boundary')
# plt.show()
#
# t1 = np.linspace(0, 14, density)
# t2 = np.linspace(0, 12, density)
#
# cordinates = [(x, y) for x in t1 for y in t2]
# # def X_to_data_frame(density, data_train):
# #     X_train = get_X(data_train)
# #     T = np.ones((density, 1))
# #     for i in range(X_train.shape[1] - 1):
# #         t1 = np.linspace(np.min(X_train[:, i + 1]) - 1, np.max(X_train[:, i + 1]) + 1, density)
# #         t1 = np.reshape(t1, (density, 1))
# #         T = np.concatenate((T, t1), axis=1)
# #     T = pd.DataFrame(T)
# #     return T
# inner_product = np.array(mapped_cord) @ np.array([-1.52363821, 1.65719443])
# # [np.abs(inner_product) < threshhold] 返回inner_product中所有小于threshhold索引
# decision = mapped_cord[np.abs(inner_product) < 0.01]


