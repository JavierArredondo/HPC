f = fopen('output/salida.raw', 'rb');
M = fread(f, 'float');
M = reshape(M,201 ,401);
imagesc(M);
colormap( [jet();flipud( jet() );0 0 0] )