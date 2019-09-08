import numpy as np
from PIL import Image

image = np.array(Image.open('pikachu.png').convert('L').resize((256, 256)))
print(image)

th = 128
image_bin = image > th

print(image_bin)

# To save image.png
im_bin_128 = (image > th) * 255
print(im_bin_128)
Image.fromarray(np.uint8(im_bin_128)).save('pikachu_bin.png')

# To sabe raw
image_raw = open("pikachu.raw", "w")
for row in image_bin:
	for el in row:
		image_raw.write("0 " if el else "1 ")
	image_raw.write("\n")
image_raw.close()