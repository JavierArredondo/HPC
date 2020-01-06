f = fopen('output/salida.raw', 'rb');
M = fread(f, 'float');
M = reshape(M,256 ,256);
imagesc(M);
axis('off');
axis('square');