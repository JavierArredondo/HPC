import numpy as np
from PIL import Image

size = 14
raw_output = "images_raw/example14x14.raw"
im_bin_128 = np.matrix([
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,1,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,1,1,1,0,0,0,0,0,0,0],
	[0,0,0,0,0,1,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,1,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,1,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,1,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	]) * 255
binarize = Image.fromarray(np.uint8(im_bin_128))
#binarize.show()
binarize = binarize.resize((512, 512))
binarize.save("example14x14.png")
# f = open(raw_output, "wb")
# byte = 0
# for i in range(size):
# 	for j in range(size):
# 		byte = int(im_bin_128.item(i, j))
# 		f.write(byte.to_bytes(4, byteorder='little'))
