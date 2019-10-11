# SIMD - Paralelismo a nivel de instrucción

## Dilatación en imágenes
Es una técnica de procesamiento de imágenes correspondiente a los operadores morfológicos, generalmente estos corresponden a un patrón bidimensional binarios o elemento de estructuración. El elemento de estructuración a utilizar es:

![ES](https://claudiovz.github.io/scipy-lecture-notes-ES/_images/diamond_kernel.png)

## Implementación
El programa se desarrolló en C y básicamente la implementación consto en utilizar la siguiente formula para aplicar el elemento de estructuración expuesto anteriormente.

<a href="https://www.codecogs.com/eqnedit.php?latex=C&space;=&space;A\oplus&space;B" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C&space;=&space;A\oplus&space;B" title="C = A\oplus B" /></a>

En otras palabras, se aplico el operador lógico `or` en cada una de las direcciones del kernel (desde el centro). Por lo cual si había a lo menos un `1` en el kernel, se marcaba el centro.

## Compilación y ejecución
Basta compilar con el makefile:

```
make
```

Ejecución, donde las distintas banderas indican:

- -i Archivo.raw de entrada
- -s Salida.raw secuencial
- -p Salida.raw con SIMD
- -N Tamaño NxN de la imagen
- -D Flag para debugear (0 no muestra nada por pantalla, 1 muestra output por pantalla)

```
./dilation.out -i ./input/lena256x256.raw -s output/lena_seq.raw -p output/lena_simd.raw -N 256 -D 0
```

Compilar y ejecutar desde makefile

```
make start
```

## Ejemplo de entrada y salida

Entrada de ejemplo 14x14 pixeles

![entrada14](https://raw.githubusercontent.com/JavierArredondo/HPC/master/Scripts/images_bin/example14x14.png?token=AEVXHRHT7WHWVVXZTI5PPS25SBQME)

Salida de ejemplo 14x14 pixeles

![salida14](https://raw.githubusercontent.com/JavierArredondo/HPC/master/Scripts/images_dilated/example14x14_seq_dilated.png?token=AEVXHRHCP3HDGAVKOMZ7PAK5SBQHU)

Entrada pikachu 256 x 256

![pikachu256e](https://raw.githubusercontent.com/JavierArredondo/HPC/master/Scripts/images_bin/pikachu_bin.png?token=AEVXHREFREUMKIZS7EF46CS5SBQQO)

Salida pikachu 256 x 256

![pikachu256s](https://raw.githubusercontent.com/JavierArredondo/HPC/master/Scripts/images_dilated/pikachu_seq_dilated.png?token=AEVXHREGL33HWKLWT4V4IOK5SBQQY)
