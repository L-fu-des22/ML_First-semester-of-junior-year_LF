# 1加载必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from PIL import Image
import cv2
from sklearn.model_selection import KFold
import random


# 读取文件

def readfile(strv, number):
    array_of_img = []
    for i in range(number):
        directory_name_1 = 'data\\Digits\\syn\\' + strv + f'\\{i}'
        for filename in os.listdir(directory_name_1):
            img = cv2.imread(directory_name_1 + "\\" + filename)
            img = cv2.resize(img, dsize=(32, 32), dst=None, fx=None, fy=None, interpolation=None)
            inf = np.array(img)
            array_of_img.append(inf)
    for i in range(number):
        directory_name_1 = 'data\\Digits\\mnist\\' + strv + f'\\{i}'
        for filename in os.listdir(directory_name_1):
            img = cv2.imread(directory_name_1 + "\\" + filename)
            img = cv2.resize(img, dsize=(32, 32), dst=None, fx=None, fy=None, interpolation=None)
            inf = np.array(img)
            array_of_img.append(inf)
    data = np.array(array_of_img)
    data = (data - data.mean())/data.std()
    #标准化
    data = np.array(data)
    print(data.shape)
    return data


def get_feature(X_train):
    return torch.from_numpy(X_train).float()


def get_label(size):
    label = []
    for i in range(10):
        for j in range(int(size / 10)):
            label.append(i)
    label = np.array(label)
    return torch.from_numpy(label)




# 5构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(3072, 300)
        self.drop2 = nn.Dropout2d(p=0.1)
        self.fc2 = nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(-1, 3072)  # 拉平，-1自动计算维度，
        x = self.drop1(x)

        x = self.fc1(x)  # 输入∶ batch* 2000输出:batch*500

        x = F.relu(x)  # 保持shpae不变
        x = self.drop2(x)
        x = self.fc2(x)  # 输入: batch*500输出:batch*10

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
        pred = output.max(1, keepdim=True)[1]  # 值，索引 # pred = torch.max ( ouput, dim=1 ) # pred = output.argmax ( dim=1 )
        # #累计正确的值
        correct += pred.eq(target.long().view_as(pred)).sum().item()
    return train_loss,correct

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
            pred = output.max(1, keepdim=True)[1]  # 值，索引 # pred = torch.max ( ouput, dim=1 ) # pred = output.argmax ( dim=1 )
            # #累计正确的值
            correct += pred.eq(target.long().view_as(pred)).sum().item()

    return test_loss,correct



if __name__ == '__main__':
    # 2定义超参数
    BATCH_SIZE = 1024 # 每批处理的数据
    print(torch.cuda.device_count())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否用GPU
    print(DEVICE)
    EPOCHS = 50  # 训练数据集的轮次



    X_train = readfile('train', 10)
    X_train = get_feature(X_train)
    X_train_size0 = X_train.shape[0]
    y_train = get_label(X_train_size0)

    X_test = readfile('test', 10)
    X_test = get_feature(X_test)
    X_train_test0 = X_test.shape[0]
    y_test = get_label(X_train_test0)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)


    dataset = ConcatDataset([train_dataset, test_dataset])
    torch.manual_seed(42)
    criterion = nn.CrossEntropyLoss()

    splits = KFold(n_splits=10, shuffle=True, random_state=42)
    foldperf = {}

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 6定义优化器
        model = Digit().to(DEVICE)
        optimizer = optim.Adam(model.parameters(),lr=0.0008)

        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

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
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            foldperf['fold{}'.format(fold + 1)] = history
            torch.save(model, 'k_cross_CNN.pt')

    testl_f, tl_f, testa_f, ta_f = [], [], [], []
    k = 10
    for f in range(1, k + 1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))
        ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
        testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))
    print('Performance of {} fold cross validation'.format(k))
    print(
        "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average "
        "Test Acc:{:.2f} ".format(np.mean(tl_f),np.mean(testl_f),np.mean(ta_f),np.mean(testa_f)))