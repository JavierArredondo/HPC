import sys
import os
import numpy as np
import pandas as pd
from PIL import Image

filename = sys.argv[1]
output = filename.split(".")[0]#filename.split("/")[-1].split(".")[0]
#df = pd.read_csv(filename, sep = " ", header = None)

file = open(filename, "rb").read()
print(file)
print("kdsakl\n")
print(len(file))
out_hex = [int('{:02X}'.format(b), 16) for b in file]

n = int(len(out_hex) ** (1/2))

out_hex = [out_hex[i::n] for i in range(n)]

image = np.array(out_hex).transpose()

a = Image.fromarray(np.uint8(image))
#a.save("images_dilated/{}.png".format(output))
a.show()