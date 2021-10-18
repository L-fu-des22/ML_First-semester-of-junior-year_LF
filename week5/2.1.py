# 1加载必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from PIL import Image
import cv2
from sklearn.model_selection import KFold
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# 读取文件

def readfile():
    array_of_img = []
    directory_name_1 = 'E:\\Junior first semester\\374-Machine Learning System Design\\week5\\homework52\\data\\Art'
    i = 0
    for filename_1 in os.listdir(directory_name_1):
        for filename_2 in os.listdir(directory_name_1 + "\\" + filename_1):
            img = cv2.imread(directory_name_1 + "\\" + filename_1 + "\\" + filename_2, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, dsize=(100, 100), dst=None, fx=None, fy=None, interpolation=None)
            inf = np.array(img).flatten()
            inf = np.append(inf, i)
            array_of_img.append(inf)
        i = i + 1
    data = np.array(array_of_img)
    # 标准化
    # data = (data - data.mean())/data.std()

    print(data.shape)
    return data


def get_feature(data):
    X_train = data[:, :-1]
    X_train = (X_train - X_train.mean()) / X_train.std()
    print(X_train.shape)
    return torch.from_numpy(X_train).float()


def get_label(data):
    label = np.array(data[:, -1])
    return torch.from_numpy(label)


# 5构建网络模型
# 0层
class Digit_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(10000, 65)

    def forward(self, x):
        x = x.view(-1, 10000)  # 拉平，-1自动计算维度，
        x = self.drop1(x)

        x = self.fc1(x)  # 输入∶ batch* 2000输出:batch*500
        x = F.relu(x)  # 保持shpae不变

        output = F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率值
        return output


# 1层
class Digit_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(10000, 2048)
        self.drop2 = nn.Dropout2d(p=0.1)
        self.fc2 = nn.Linear(2048, 65)

    def forward(self, x):
        x = x.view(-1, 10000)  # 拉平，-1自动计算维度，
        x = self.drop1(x)

        x = self.fc1(x)  # 输入∶ batch* 2000输出:batch*500

        x = F.relu(x)  # 保持shpae不变
        x = self.drop2(x)
        x = self.fc2(x)  # 输入: batch*500输出:batch*10

        output = F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率值
        return output


# 2层
class Digit_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(10000, 2048)
        self.drop2 = nn.Dropout2d(p=0.1)
        self.fc2 = nn.Linear(2048, 512)
        self.drop3 = nn.Dropout2d(p=0.1)
        self.fc3 = nn.Linear(512, 65)

    def forward(self, x):
        x = x.view(-1, 10000)  # 拉平，-1自动计算维度，

        x = self.drop1(x)
        x = self.fc1(x)  # 输入∶ batch* 2000输出:batch*500
        x = F.relu(x)  # 保持shpae不变

        x = self.drop2(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.drop3(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率值
        return output


# 7定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    correct = 0.0
    # 测试损失
    train_loss = 0.0
    # 模型训练
    model.train()
    for batch_index, (data, target) in (enumerate(train_loader)):
        # 部署到DEVICE上去
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target.long())

        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        train_loss += loss.item()

        train_loss += F.cross_entropy(output, target.long()).item()
        # 找到概率值最大的下标
        pred = output.max(1, keepdim=True)[
            1]  # 值，索引 # pred = torch.max ( ouput, dim=1 ) # pred = output.argmax ( dim=1 )
        # #累计正确的值
        correct += pred.eq(target.long().view_as(pred)).sum().item()
    return train_loss, correct


# 8定义测试方法
def test_mode1(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():  # 不会计算梯度，也不会进行反向传播
        for data, target in test_loader:
            # 部署到device上
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target.long()).item()
            # 找到概率值最大的下标
            pred = output.max(1, keepdim=True)[
                1]  # 值，索引 # pred = torch.max ( ouput, dim=1 ) # pred = output.argmax ( dim=1 )
            # #累计正确的值
            correct += pred.eq(target.long().view_as(pred)).sum().item()

    return test_loss, correct


if __name__ == '__main__':
    # 2定义超参数
    BATCH_SIZE = 512  # 每批处理的数据
    print(torch.cuda.device_count())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否用GPU
    print(DEVICE)
    EPOCHS = 80  # 训练数据集的轮次

    data = readfile()
    X_train = get_feature(data)

    y_train = get_label(data)

    dataset = TensorDataset(X_train, y_train)

    torch.manual_seed(42)

    splits = KFold(n_splits=10, shuffle=True, random_state=42)
    foldperf = {}
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    hidden_num = 2
    lamb = 0.01

    hidden_num_label = ''
    lamb_label = ''


    Train_loss = np.zeros((10, EPOCHS))  # 储存Train_loss数据，行为fold，列为对应Epoch的loss值。
    Test_loss = np.zeros((10, EPOCHS))
    Train_acc = np.zeros((10, EPOCHS))
    Test_acc = np.zeros((10, EPOCHS))
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print(f'Fold: {fold + 1},hidden_num:{hidden_num},lamb:{lamb}')

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 6定义优化器
        model = Digit_0().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=0)

        if hidden_num == 0:
            model = Digit_0().to(DEVICE)
            hidden_num_label = '[65536,65]'
        elif hidden_num == 1:
            model = Digit_1().to(DEVICE)
            hidden_num_label = '[65536,2048,65]'
        elif hidden_num == 2:
            model = Digit_2().to(DEVICE)
            hidden_num_label = '[65536,2048,512,65]'

        if lamb == 0:
            optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=0)
            lamb_label = '0'
        elif lamb == 0.01:
            optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.01)
            lamb_label = '0.01'
        elif lamb == 0.0001:
            optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.0001)
            lamb_label = '0.0001'
        elif lamb == 1:
            optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.0001)
            lamb_label = '1'
        for epoch in range(EPOCHS):
            train_loss, train_correct = train_model(model, device, train_loader, optimizer, epoch)
            test_loss, test_correct = test_mode1(model, device, test_loader)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100

            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100
            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} "
                  "AVG Training Acc {:.2f} ""% AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                             EPOCHS,
                                                                             train_loss,
                                                                             test_loss,
                                                                             train_acc,
                                                                             test_acc))
            Train_loss[fold, epoch] = train_loss
            Test_loss[fold, epoch] = test_loss
            Train_acc[fold, epoch] = train_acc
            Test_acc[fold, epoch] = test_acc

    plt.figure()
    ax = plt.subplot(111)
    ax.set_clip_on(False)

    # 以测试集准确率的收敛后的最后五个值之和为标准，选出最好、中、最差的
    Test_acc_totol_epoch = Test_acc[:, [-5, -4, -3, -2, -1]].sum(axis=1)
    print(Test_acc)
    print(Test_acc_totol_epoch)

    # 最好的
    Test_acc_maxx_index = np.where(Test_acc_totol_epoch == np.max(Test_acc_totol_epoch))
    Test_acc_maxx_index = np.array(Test_acc_maxx_index)[0, 0]
    print(Test_acc_maxx_index)
    Train_loss_maxx = Train_loss[Test_acc_maxx_index, :]
    Test_loss_maxx = Test_loss[Test_acc_maxx_index, :]

    # 最差的
    Test_acc_minn_index = np.where(Test_acc_totol_epoch == np.min(Test_acc_totol_epoch))
    Test_acc_minn_index = np.array(Test_acc_minn_index)[0,0]
    print(Test_acc_minn_index)
    Train_loss_minn = Train_loss[Test_acc_minn_index, :]
    Test_loss_minn = Test_loss[Test_acc_minn_index, :]

    # 画折线图
    x = np.arange(1, EPOCHS + 1, 1)
    # 表现最好的loss
    l_train_maxx = plt.plot(x, Train_loss_maxx.flatten().reshape((len(x), 1)), 'go-',
                            label=f'train_best:Fold{Test_acc_maxx_index+1}')
    l_test_maxx = plt.plot(x, Test_loss_maxx.flatten().reshape((len(x), 1)), 'g^-',
                           label=f'test_best:Fold{Test_acc_maxx_index+1}')

    # 表现最差的loss
    l_train_minn = plt.plot(x, Train_loss_minn.flatten().reshape((len(x), 1)), 'ro-',
                            label=f'train_worst:Fold{Test_acc_minn_index+1}')
    l_test_minn = plt.plot(x, Test_loss_minn.flatten().reshape((len(x), 1)), 'r^-',
                           label=f'test_worst:Fold{Test_acc_minn_index+1}')

    acc_train_convege_best = Train_acc[Test_acc_maxx_index, [-5, -4, -3, -2, -1]].mean()
    acc_train_convege_best = round(acc_train_convege_best,3)
    acc_test_convege_best = Test_acc[Test_acc_maxx_index, [-5, -4, -3, -2, -1]].mean()
    acc_test_convege_best = round(acc_test_convege_best, 3)


    acc_train_convege_worst = Train_acc[Test_acc_minn_index, [-5, -4, -3, -2, -1]].mean()
    acc_train_convege_worst = round(acc_train_convege_worst, 3)
    acc_test_convege_worst = Test_acc[Test_acc_minn_index, [-5, -4, -3, -2, -1]].mean()
    acc_test_convege_worst = round(acc_test_convege_worst, 3)

    # round(a, 2) = 12.35
    plt.title('best\worst Fold')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()

    # Train_acc_mean = Train_acc.mean()
    # Test_acc_mean = Test_acc.mean()
    y_axis_max = Test_loss_maxx.max()
    ax.annotate(f'layer={hidden_num_label}', (7,y_axis_max), fontsize="x-small", annotation_clip=False)
    ax.annotate(f'lambda={lamb_label}', (7, y_axis_max*0.98), fontsize="x-small", annotation_clip=False)
    ax.annotate(f'acc_train_convege_best={acc_train_convege_best}%', (7, y_axis_max*0.96), fontsize="x-small", annotation_clip=False)
    ax.annotate(f'acc_test_convege_best={acc_test_convege_best}%', (7, y_axis_max*0.94), fontsize="x-small", annotation_clip=False)
    ax.annotate(f'acc_train_convege_worst={acc_train_convege_worst}%', (7, y_axis_max*0.92), fontsize="x-small", annotation_clip=False)
    ax.annotate(f'acc_test_convege_worst={acc_test_convege_worst}%', (7, y_axis_max*0.9), fontsize="x-small", annotation_clip=False)
    plt.show()
    #                 history['train_loss'].append(train_loss)
    #                 history['test_loss'].append(test_loss)
    #                 history['train_acc'].append(train_acc)
    #                 history['test_acc'].append(test_acc)
    #                 foldperf['fold{}'.format(fold + 1)] = history
    #                 torch.save(model, 'k_cross_CNN.pt')
    #
    # print(history)
    # testl_f, tl_f, testa_f, ta_f = [], [], [], []
    # k = 10
    # for f in range(1, k + 1):
    #     tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
    #     testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))
    #     ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
    #     testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))
    # print('Performance of {} fold cross validation'.format(k))
    # print(
    #     "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average "
    #     "Test Acc:{:.2f} ".format(np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f)))
