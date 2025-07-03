#先导入模块
import torch
import torch.nn as nn  #神经网络模块，提供多种神经网络层
import torch.nn.functional as F  #提供函数式写法的层，比如F.relu(x)???什么鬼，一会问一下对比一块看

#定义模型类
    #这里的nn.Module是基类的module，我们先用的MLP需要先继承它
class MLP(nn.Module):  # 三层的前馈神经网络（无卷积）
    def _init_(self):
        super()._init_() #这里的初始化也是先调用父类nn.Module的构造函数
        self.flatten = nn.Flatten()   #把输入的2D图像展平为一维,因为神经网络只处理向量
        self.fc1 = nn.Linear(28*28,128)  #全连接层：输入784，输出128个神经元
        self.fc2 = nn.Linear(128,64)   #第二层：输入128，输出64
        self.output = nn.Linear(64,10) #输出层：输出10个神经元，对应数字0-9

    #定义前向传播函数
    def forward(self,x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))  # F.relu是RelU激活函数，用于让网络非线性，处理复杂关系
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x