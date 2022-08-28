import csv
import os

img_path = r"D:\Python\source\fire\images\val"
label_path = r"D:\Python\source\fire\labels\val"
for img_name in os.listdir(img_path):
    if img_name.replace("jpg", "txt") not in os.listdir(label_path):
        print(img_name)
print("one-one")
