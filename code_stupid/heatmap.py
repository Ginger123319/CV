# 下面代码实现了SHAP图和热力图的生成、展示和保存功能：

import torch
import shap
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms

# 加载模型
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load(r'D:\Python\pretrained_weights\checkpoints\resnet18-f37072fd.pth'))
model.eval()

# 加载图像预处理函数
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载图像
img_path = r'D:\Python\test_jzyj\key_mouse_20\pic\0001.jpg'
img = Image.open(img_path)
img = preprocess(img)

# 创建 SHAP 解释器
explainer = shap.Explainer(model, preprocess)

# 计算 SHAP 值
shap_values = explainer(img)

# 绘制 SHAP 图
shap.image_plot(shap_values, img.permute(1, 2, 0))
plt.savefig('output/shap.png')
plt.show()

# 计算热力图
heatmap = torch.abs(shap_values).sum(dim=1).squeeze()

# 绘制热力图
plt.imshow(heatmap.detach().numpy(), cmap='gray')
plt.savefig('output/heatmap.png')
plt.show()
