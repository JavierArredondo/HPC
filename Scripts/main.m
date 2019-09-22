f = fopen('images_raw/circulos.raw', 'r');
s = fread(f, 'int'); fclose(f);
s = reshape(s, 256, 256);
s = s';
imagesc(s); axis('square'); colormap(gray); axis(off)
