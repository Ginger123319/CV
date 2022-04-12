from PIL import Image,ImageDraw

img = Image.open("bg_pic/1.jpg")
draw = ImageDraw.Draw(img)
draw.rectangle((447,172,536,359),outline="red",width=2)
img.show()