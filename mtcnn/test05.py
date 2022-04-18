from PIL import Image,ImageDraw
import os

img = Image.open("data/000002.jpg")
draw = ImageDraw.Draw(img)
draw.rectangle(( 72  ,94 ,72+221,94+ 306),outline="red",width=2)
img.show()

