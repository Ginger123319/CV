from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torch

if __name__ == '__main__':
    # print(maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False))
    net = maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, trainable_backbone_layers=5)
    print(net.backbone)
    net.eval()
    data = torch.randn(1, 3, 28, 28)
    print(net(data))
