```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import matplotlib.pyplot as plt
# import tensorflow as tf
from sklearn.metrics import classification_report#这个包是评价报告
import scipy.optimize as opt

# 把Excel文件中的数据读入pandas
data = pd.read_excel(r'E:\Junior first semester\374-Machine Learning System Design\week3\Homework3\Homework3\data\Disease\data_1.xlsx',)

sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
sns.lmplot('BP_HIGH', 'BMI', hue='HYPERTENTION', data=data,
           size=6,
           fit_reg=False,
           scatter_kws={"s": 50}
          )
plt.show()

def get_X(df):#读取特征
#     """
#     use concat to add intersect feature to avoid side effect
#     not efficient for big dataset though
#     返回data是个2维矩阵
#     """
    ones = pd.DataFrame({'ones': np.ones(len(df))})#ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return np.array(data.iloc[:, :-1]) # 这个操作返回 ndarray,不是矩阵

def get_y(df):#读取标签
#     '''
#     assume the last column is the target
#     返回y是一维矩阵
#     '''
    return np.array(df.iloc[:, -1])#df.iloc[:, -1]是指df的最后一列

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)

def gradient(theta, X, y):
#     '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

def feature_mapping(x, y, power, as_ndarray=False):
#     """return mapped features as ndarray or dataframe"""
    # data = {}
    # # inclusive
    # for i in np.arange(power + 1):
    #     for p in np.arange(i + 1):
    #         data["f{}{}".format(i - p, p)] = np.power(x, i - p) * np.power(y, p)

    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
            }

    if as_ndarray:
        return np.array(pd.DataFrame(data))
    else:
        return pd.DataFrame(data)

x1 = np.array(data['BP_HIGH'])

x2 = np.array(data['BMI'])

data_ft = feature_mapping(x1, x2, power=3)

theta = np.zeros(data_ft.shape[1])

X = feature_mapping(x1, x2, power=3, as_ndarray=True)

y = get_y(data)

def regularized_cost(theta, X, y, l=1):
#     '''you don't penalize theta_0'''
    theta_j1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

    return cost(theta, X, y) + regularized_term #正则化代价函数

def regularized_gradient(theta, X, y, l=1):
#     '''still, leave theta_0 alone'''
# regularized gradient(正则化梯度)
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term

#使用不同的  𝜆  （这个是常数）
#画出决策边界

def draw_boundary(power, l):
#     """
#     power: polynomial power for mapped feature
#     l: lambda constant
#     """
    density = 300
    threshhold = 0.01

    final_theta = feature_mapped_logistic_regression(power, l)
    x, y = find_decision_boundary(density, power, final_theta, threshhold)
    df = pd.read_excel(r'E:\Junior first semester\374-Machine Learning System Design\week3\Homework3\Homework3\data\Disease\data_1.xlsx',)

    sns.lmplot('BP_HIGH', 'BMI', hue='HYPERTENTION', data=df, size=6, fit_reg=False, scatter_kws={"s": 100})

    plt.scatter(x, y, c='Red', s=1)
    plt.title('Decision boundary')
    plt.show()

def feature_mapped_logistic_regression(power, l):
#     """for drawing purpose only.. not a well generealize logistic regression
#     power: int
#         raise x1, x2 to polynomial power
#     l: int
#         lambda constant for regularization term
#     """
    df = pd.read_excel(r'E:\Junior first semester\374-Machine Learning System Design\week3\Homework3\Homework3\data\Disease\data_1.xlsx', )

    x1 = np.array(df.BP_HIGH)
    x2 = np.array(df.BMI)
    y = get_y(df)

    X = feature_mapping(x1, x2, power, as_ndarray=True)
    theta = np.zeros(X.shape[1])

    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient)

    final_theta = res.x
    print('The solution of the optimization is ')
    print(final_theta)
    # 训练集预测模型报告
    y_pred = predict(X, final_theta)
    print('训练集预测报告：')
    print(classification_report(y, y_pred))

    # 读取测试集，并测试训练集生成的模型
    df_test = pd.read_excel(r'E:\Junior first semester\374-Machine Learning System Design\week3\Homework3\Homework3\data\Disease\data_1.xlsx', sheet_name='test')

    x1_test = np.array(df_test.BP_HIGH)
    x2_test = np.array(df_test.BMI)
    y_test = get_y(df_test)
    X_test = feature_mapping(x1_test, x2_test, power, as_ndarray=True)

    y_test_pred = predict(X_test, final_theta)
    print('测试集预测报告：')
    print(classification_report(y_test, y_test_pred))
    return final_theta

#寻找决策边界函数
def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(100, 210, density)
    t2 = np.linspace(10, 40, density)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power)  # this is a dataframe
    inner_product = np.array(mapped_cord) @ theta
    # [np.abs(inner_product) < threshhold] 返回inner_product中所有小于threshhold索引
    decision = mapped_cord[np.abs(inner_product) < threshhold]

    return decision.f10, decision.f01

draw_boundary(power=3, l=5)#lambda=惩罚系数
```