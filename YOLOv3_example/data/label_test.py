from PIL import Image,ImageDraw
img=Image.open('images/01.jpg')
draw=ImageDraw.Draw(img)
draw.rectangle((45,93,282,344),width=3)
draw.rectangle((205,71,375,339),width=3)
img.show()
