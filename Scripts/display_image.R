setwd("HPC/Scripts/")

imagen = as.matrix(read.table(file = "pikachu.raw"))

image(x = imagen, col = c("white", "black"), axes = F)


write(x = imagen, file = "imagen.raw", ncolumns = 256, sep = "")
