from collections import OrderedDict
from torchvision import models, transforms
from PIL import Image
import torch
from torch.nn import functional as F
import numpy as np

transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])

with open('imagenet_classes_1k.txt') as f:
    classes = [line.strip() for line in f.readlines()]
classes = np.array(classes)

net = models.resnet50(pretrained=False)
checkpoint = torch.load(r'D:\download\rn50_relabel_cutmix_IN21k_81.2.pth')
new_state_dict = OrderedDict([])
for k, v in checkpoint.items():
    new_state_dict[k[7:]] = v
net.load_state_dict(new_state_dict)

net.eval()
# exit()
img = Image.open(r'D:\Python\source\VOCdevkit\voc2012-500\img\2007_000629.jpg')
img = img.convert('RGB')

img = transform(img)
img = img.unsqueeze(dim=0)

result = net(img)
_, index = torch.max(result, 1)

multi_result = torch.sigmoid(result)
print(multi_result > 0.5)
print(classes[(multi_result > 0.5)[0]])
# exit()

percentage = F.softmax(result, dim=1)[0] * 100

print(classes[index[0]], percentage[index[0]].item())

_, indices = torch.sort(result, descending=True)
[print((classes[idx], percentage[idx].item())) for idx in indices[0][:5]]
