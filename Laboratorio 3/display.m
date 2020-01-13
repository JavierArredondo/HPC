f = fopen('output/salida_1.raw', 'rb');
M = fread(f, 'float');
M = reshape(M,256 ,256);
imagesc(M);
axis('off');
axis('square');