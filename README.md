# 
残差神经网络在手写签名识别中的应用

1.首先导入相应的包：
import os
import torch 
from torch import nn,optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets,transforms

2.以下为图像数据预处理操作：
train_path = "./pytorch_datasets/train"
test_path = "./pytorch_datasets/test"
定义数据集预处理的方法
data_transform = transforms.Compose([
#        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(150),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
])

datasets_train = datasets.ImageFolder(train_path,data_transform)
datasets_test = datasets.ImageFolder(test_path,data_transform)

train_loader = data.DataLoader(datasets_train,batch_size=32,shuffle=True)
test_loader = data.DataLoader(datasets_test,batch_size=16,shuffle=False)

3.再通过上述提到的的pytorch搭建卷积神经网络：
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
model = CNN().to(DEVICE)

4.对图像数据集进行训练，打印30次损失
for epoch in range(30):
    model.train()
    for i,(image,label) in enumerate(train_loader):
        data,target = Variable(image).cuda(),Variable(label.cuda()).unsqueeze(-1)
        optimizer.zero_grad()
        output = model(data)
        output=output.to(torch.float32)
        target=target.to(torch.float32)
#         print(output.shape,target.shape)
        loss = F.binary_cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        if (i+1)%10==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (i+1) * len(data), len(train_loader.dataset),
            100. * (i+1) / len(train_loader), loss.item()))
以下为30次训练结果：

5.对测试集数据进行验证评估，打印精确度：
for epoch in range(30):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
       for data, target in test_loader:
          data, target = data.to(DEVICE), target.to(DEVICE).float().unsqueeze(-1)
          output = model(data)
          test_loss += F.binary_cross_entropy(output, target, reduction='sum').item() # 将一批的损失相加
            pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(DEVICE)
            correct += pred.eq(target.long()).sum().item()
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
下图为精确度打印结果：

6.模型训练
6.1 数据提取
本系统在数据提取时候，是创建一个映射，在递归中用os来打开文件，并且将文件使用torchvision包中的transform方法进行加强裁剪和处理图像。并将训练所需的数据通过dataset和dataloder加载如模型。


6.2 模型搭建
本系统运用ResNet18来作为训练所需的模型。由何凯明团队所提出来的。其深度残差网络（Deep Residual Network）在2015年的ImageNet上取得冠军。该模型可以很好地解决深度递增而造成的退化问题。
具体代码：import torch
from torch import nn
from torch.nn import functional as F

class ResBlk(nn.Module):
    #resnet block

    def __init__(self, ch_in, ch_out, stride=1):
    super(ResBlk, self).__init__()

    self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)#设置卷积层
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, num_class):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )

        self.blk1 = ResBlk(16, 32, stride=3)
        self.blk2 = ResBlk(32, 64, stride=3)
        self.blk3 = ResBlk(64, 128, stride=2)
        self.blk4 = ResBlk(128, 256, stride=2)

        self.outlayer = nn.Linear(256*3*3, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))#激励函数开始去除负值
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x
6.3 模型训练
在文件中，通过对手写签名的字体集合进行训练，得到无拟合问题的训练模型。同时可以看到，模型的准确率无线接近100%，同时loss也降到0左右。

将训练获得的训练模型装载，并系统的使用其进行签名的识别。这里笔者结合着计算机视觉常用的库opencv进行使用模型。并且语音播报签名人，并且打印正确错误信息。
