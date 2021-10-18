# 1加载必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from PIL import Image
import cv2

# 2定义超参数
BATCH_SIZE = 64  # 每批处理的数据
print(torch.cuda.device_count())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否用GPU
EPOCHS = 5  # 训练数据集的轮次


# 读取文件

def readfile(strv, number):
    array_of_img = []
    for i in range(number):
        directory_name_1 = 'data\\Digits\\mnist\\' + strv + f'\\{i}'
        for filename in os.listdir(directory_name_1):
            img = cv2.imread(directory_name_1 + "\\" + filename)  # , cv2.IMREAD_GRAYSCALE
            img = cv2.resize(img, dsize=(32, 32), dst=None, fx=None, fy=None, interpolation=None)
            inf = np.array(img)
            array_of_img.append(inf)
    data = np.array(array_of_img)
    print(data.shape)
    data = (data - data.mean()) / data.std()
    # 标准化
    data = np.array(data)
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


X_train = readfile('train', 10)
X_train = get_feature(X_train)
X_train_size0 = X_train.shape[0]
y_train = get_label(X_train_size0)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

X_test = readfile('test', 10)
X_test = get_feature(X_test)
X_test_size0 = X_test.shape[0]
y_test = get_label(X_test_size0)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # 拉平，-1自动计算维度，
        x = self.fc1(x)  # 第一层
        x = F.relu(x)  # 第一层激活函数

        x = self.fc2(x)  # 输出层
        output = F.log_softmax(x, dim=1)  # 输出层激活函数，计算分类后，每个数字的概率值
        return output


# 6定义优化器
model = Digit().to(DEVICE)
optimizer = optim.Adam(model.parameters())


# 7定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
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
        # 找到概率值最大的下标
        pred = output.max(1, keepdim=True)  # pred = output.argmax(dim=1)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))


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
            pred = output.max(1, keepdim=True)[1]  # 值，索引
            # pred = torch.max ( ouput, dim=1 )
            # pred = output.argmax ( dim=1 )
            # #累计正确的值
            correct += pred.eq(target.long().view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test ——Average loss : {:.4f},Accuracy : {:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)))


# 9调用方法7/ 8
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_mode1(model, DEVICE, test_loader)
