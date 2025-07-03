import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MLP
from train import train
from test import test

def main():
    # 1.设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2.超参数设置
    batch_size = 64
    learning_rate = 0.01
    epochs = 5

    # 3.数据预处理 & 数据集加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4.模型，损失函数，优化器实例化
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #5. 训练和测试循环
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)     #这几个参数是怎么知道要填这几个的，我们在写train和test的时候也没说啊

    # 6. 保存模型
    torch.save(model.state_dict(), "model.pth")
    print("模型已保存到 model.pth")

if __name__ == "__main__":
    main()

