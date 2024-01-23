import torch
import torchvision
import numpy as np

list1 = [1,2,3]
list1 = np.array(list1)
list2 = list1.view()
list3 = np.copy(list1)
list1[0] = 0
print(id(list2.base))
print(type(list3))
print(list1[0])
print(id(list1))
print(id(list2))
print(id(list3))

exit()

pretrained_pth = "model_dict_path"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载匹配的权重，也可以打印没有加载到的权重的key
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
model_dict = model.state_dict()
pretrained_dict = torch.load(pretrained_pth, map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)