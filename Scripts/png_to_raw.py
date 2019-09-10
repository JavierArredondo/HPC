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
	raw_output = "images_raw/{}.raw".format(filename)
	image = Image.open("{}/{}".format(work_directory, f))
	image_array = np.array(image.convert('L').resize((size, size)))
	th = 100
	image_bin = image_array > th
	im_bin_128 = (image_array > th) * 255
	binarize = Image.fromarray(np.uint8(im_bin_128))
	# Display result
	binarize.save(bin_output)

	# To save raw
	image_raw = open(raw_output, "w")
	for row in image_bin:
		for el in row:
			image_raw.write("1 " if el else "0 ")
		image_raw.write("\n")
	image_raw.close()