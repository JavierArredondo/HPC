import numpy as np
from PIL import Image
import pandas as pd

filename = "../Laboratorio 1/output/pikachu_sdilation.raw"

df = pd.read_csv(filename, sep = " ", header = None)

print(df.head)

image = np.array(df) * 255
print(np.uint8(image))

a = Image.fromarray(np.uint8(image))
a.show()
# To sabe raw
image_raw = open("pikachu.raw", "w")
for row in image_bin:
	for el in row:
		image_raw.write("0 " if el else "1 ")
	image_raw.write("\n")
image_raw.close()
