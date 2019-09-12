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
	raw_output = "images_raw/{}sss.RAW".format(filename)
	image = Image.open("{}/{}".format(work_directory, f))
	image_array = np.array(image.convert('L').resize((size, size)))
	th = 100
	image_bin = image_array > th
	im_bin_128 = (image_array > th) * 255
	binarize = Image.fromarray(np.uint8(im_bin_128))
	binarize.show()
	break
	# Display result
	binarize.save(bin_output, "RAW")

