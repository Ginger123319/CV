import torch
from torch import nn
from torchvision import models


class Resnet50Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        net = models.resnet50(pretrained=False)
        net.load_state_dict(torch.load(
            r'C:\Users\Ginger\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth'))
        for param in net.parameters():  # 冻结参数
            param.requires_grad_(False)
        features_tmp = nn.Sequential(*list(net.children())[:-1])
        features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.features = nn.Sequential(*list(features_tmp))
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        feature = self.features(x)
        x = feature.view(feature.size(0), -1)
        output = self.classifier(x)
        return output


def _get_model(model_type, num_class):
    print("要构造的模型：{}".format(model_type))
    if model_type == "Resnet50":
        return Resnet50Net(num_classes=num_class)
    else:
        raise Exception("不支持这种模型：{}".format(model_type))


if __name__ == '__main__':
    model_type = "Resnet50"
    class_num = 20
    model = _get_model(model_type, class_num)
    print(type(model))
    data = torch.randn(3, 3, 256, 256)
    print(model(data).shape)
