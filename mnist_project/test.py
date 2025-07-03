import torch
from dataset import get_dataloader
train_loader, test_loader = get_dataloader(batch_size=64)
from model import MLP

def test(model, device):
    # 4. 加载模型并回复权重
    model.load_state_dict(torch.load("saved_model.pth", map_location=device))# 这是干啥了：把这些参数加载到你定义的模型结构中。
    model.eval()  # 推理模式

    # 5. 推理阶段不要计算梯度
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, dim=1)  # 这是啥：
            # output 是一个 [batch_size, 10] 的 tensor，表示每张图对 0~9 每个数字的预测分数（logits）。
            # torch.max(output, dim=1) 会返回：第一个是最大值，第二个是最大值对应的索引
            # 我们要的就是索引，所以逗号前是_，表示啥也没,后面是predicated，也就是我们预测的索引，在这里就是数字
            correct += (predicted == target).sum().item()  # 为什么要这样写
            # predicted == target返回一个布尔张量，比如[True, False, True, False]
            # .sum()会把这个张量里的True视为1，False视为0，然后相加
            # .item()把sum加起来的结果变为py里的int，可以累加到correct里
            total += target.size(0)

    # 6. 输出最终准确率
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    test(model, device)
