f = fopen('output/salida.raw', 'rb');
M = fread(f, 'float');
M = reshape(M,2001 ,2001);
imagesc(M);
colormap( [jet();flipud( jet() );0 0 0] )