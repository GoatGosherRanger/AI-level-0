import torch
import torch.nn as nn
import torch.optim as optim     #这个是选择优化器?所以选择优化器是啥
from dataset import get_dataloader
train_loader, test_loader = get_dataloader(batch_size=64)
from model import MLP


def train(model, device, train_loader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):  #训练5个轮次
        model.train()  #启用训练模式
        running_loss = 0.0  #？？这是啥 这是统计整个 epoch 的累计损失。最后每隔100个求个平均，看看是不是在下降

        for batch_idx,(data,targets) in enumerate(train_loader):
            data,targets = data.to(device),targets.to(device)    #这data和targets是什么东西 :data就是图片，targets是标签

            optimizer.zero_grad()   #清空梯度
            outputs = model(data)   #正向传播         为什么可以直接在括号里放data，这是执行了啥了  PyTorch 模型的语法糖，默认是model.forward(x)
            loss = criterion(outputs,targets)   #计算损失
            loss.backward()
            optimizer.step()        #更新参数

            running_loss += loss.item()

            # 每隔100个batch打印一次平均Loss
            if batch_idx %100 ==0:
               print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")  #这是干啥了？:每隔 100 个 batch 打印一次当前 loss，方便你实时观察训练情况。

    #保存训练好的模型
    torch.save(model.state_dict(),"saved_model.pth")
    print("**************************************************Training finished. Model saved as 'saved_model.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)  #新建一个神经网络对象，把它放到device上
    criterion = nn.CrossEntropyLoss() #库里提供的多分类问题的损失函数
    optimizer = optim.Adam(model.parameters(),lr = 0.001) #一个优化器，学习率为0.001

    train(model, device, train_loader, optimizer, criterion, epochs=5)
