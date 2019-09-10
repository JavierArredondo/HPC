import sys
import os
import numpy as np
import pandas as pd
from PIL import Image

filename = sys.argv[1]
output = filename.split("/")[-1].split(".")[0]
df = pd.read_csv(filename, sep = " ", header = None)
image = np.array(df) * 255
a = Image.fromarray(np.uint8(image))
a.save("images_dilated/{}.png".format(output))
#a.show()