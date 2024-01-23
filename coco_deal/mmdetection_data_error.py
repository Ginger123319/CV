from pycocotools.coco import COCO
import json

coco = COCO('data/coco/annotations/instances_train2017-2.json')
imgIds = coco.getImgIds()
imgs = coco.loadImgs(imgIds)
anns = coco.loadAnns(coco.getAnnIds())
cats = coco.loadCats(coco.getCatIds())

for ann in anns:
    del ann['ignore']
for img in imgs:
    del img['license']
    del img['flickr_url']
    del img['coco_url']
label = {'images': imgs, 'annotations': anns, 'categories': cats}
with open('instances_train2017.json', 'w') as f:
    json.dump(label, f)
