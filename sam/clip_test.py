from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel  # 挂梯子能下载

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # openai/clip-vit-base-patch32

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image = Image.open('liudehua.jpeg')
text = ["Jackie Chan", "Andy Lau", 'Chow Yun Fat']

inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)

for i in range(len(text)):
    print(text[i], ':', probs[0][i])