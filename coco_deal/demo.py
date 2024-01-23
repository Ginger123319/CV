import os
from PIL import Image
from PIL import ImageDraw

img_path = r'D:\download\archive\ships-aerial-images\valid\images'
label_path = r'D:\download\archive\ships-aerial-images\valid\labels'
json_path = r'D:\python\source\语义分割json标准目录\annotation.json'
from pycocotools.coco import COCO

coco = COCO(json_path)
print(coco.getCatIds())
print([cat['name'] for cat in coco.loadCats(coco.getCatIds())])
exit()

print(len(os.listdir(img_path)), len(os.listdir(label_path)))

for img, label in zip(os.listdir(img_path), os.listdir(label_path)):
    print(img, label)
    img = Image.open(os.path.join(img_path, img)).convert('RGB')
    w, h = img.size
    print(w, h)

    draw = ImageDraw.Draw(img)
    with open(os.path.join(label_path, label), 'r') as f:
        for bbox in f.readlines():
            bbox = list(map(float, bbox.split()))
            bbox.pop(0)
            bbox[0] *= w
            bbox[1] *= h
            bbox[2] *= w
            bbox[3] *= h
            draw.point((bbox[0], bbox[1]), "#FF0000")
            img.show()

            print(bbox)
            exit()
