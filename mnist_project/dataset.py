#流程 train_dataset → train_loader → batch of (images, labels) → 模型训练
import torch
from torchvision import datasets,transforms
from torch.untils.data import DataLoader

def get_dataloader(batch_size =64):        #batch_size 是干嘛的 --batch 是组的意思，=64意味着一次喂64个图片
    transform = transforms.Compose([
        transforms.ToTenser(),             #把图片进行格式转化，转化位Tenser，同时把像素值从0225映射到01
        transforms.Normalize((0.1307,),(0.3081,))    #对Tenser进行归一化，把像素值变为（x - mean（均值））/std（方差），这样处理后均值为0，方差为1
    ])
    # 加载数据集，存在./data里，并进行transform
    train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    test_dataset  = datasets.MNIST(root='./data',train=False,download=True,transform=transform)
    # 创建Loader，其中shuffle是打乱
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader, test_loader

