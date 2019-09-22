import sys
import os
import numpy as np
from PIL import Image

def raw_to_png(input_bytes):
	input_bytes = input_bytes.read()
	image_array = np.array([int(input_bytes[byte]) for byte in range(0, len(input_bytes), 4)])
	size = int(len(image_array)**(1/2))
	image_matrix = image_array.reshape(size, size)
	image = np.array(image_matrix)#.transpose()
	image = Image.fromarray(np.uint8(image))
	image = image.resize((512, 512))	
	return image

work_directory = sys.argv[1]
files = os.listdir(work_directory)
for file in files:
	filename = file.split(".")[0]
	output_filename = "images_dilated/{}_dilated.png".format(filename)
	file_tmp = open("{}/{}".format(work_directory, file), "rb")
	image = raw_to_png(file_tmp)
	file_tmp.close()
	image.save(output_filename)
#image.save("images_dilated/{}.png".format(output))