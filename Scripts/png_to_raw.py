import sys
import os
import numpy as np
from PIL import Image

size = int(sys.argv[1])
work_directory = "images_png"
files = os.listdir(work_directory)

for f in files:
	filename = f.split(".")[0]
	bin_output = "images_bin/{}_bin.png".format(filename)
	raw_output = "images_raw/{}256x256.raw".format(filename)
	image = Image.open("{}/{}".format(work_directory, f))
	image_array = np.array(image.convert('L').resize((size, size)))
	th = 100
	image_bin = image_array > th
	im_bin_128 = (image_array > th) * 255
	im_bin_128[im_bin_128 == 0] = 1
	im_bin_128[im_bin_128 == 255] = 0
	im_bin_128[im_bin_128 == 1] = 255
	binarize = Image.fromarray(np.uint8(im_bin_128))
	binarize.save("{}/{}_bin.png".format("images_bin", filename))
	f = open(raw_output, "wb")
	byte = 0
	for i in range(size):
		for j in range(size):
			byte = int(im_bin_128.item(i, j))
			f.write(byte.to_bytes(4, byteorder='little'))
	f.close()
