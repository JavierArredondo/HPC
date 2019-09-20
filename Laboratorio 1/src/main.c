# include "image.c"

int main(int argc, char *argv[])
{
	/*Parameters of the program: 
	-i     Input image. e.g: input/lena.raw
	-s:    Output image of secuential's process. e.g: output/lena_s_dilation.raw 
	-p:    Output image of simd's process.
	-N:    Weight of the input image. e.g: 256
	-D:    Debug option. If you want debug set 1, else 0.
	*/
	opterr = 0;
	if (argc < 11){
		printf("The number of parameters less than requested\n");
		return 1;
	}
	else if(argc > 11){
		printf("The number of parameters greater than the requested\n");
		return 1;
	}
	/*Get values with getopt*/ 
	int c, size, debug;
	char* input_raw = (char*) malloc(sizeof(char) * 50);
	char* output_sequential = (char*) malloc(sizeof(char) * 50);
	char* output_simd = (char*) malloc(sizeof(char) * 50);
	while ((c = getopt (argc, argv, "i:s:p:N:D:")) != -1)
	switch (c) {
		case 'N':
			sscanf(optarg, "%d", &size);
			break;
		case 'D':
			sscanf(optarg, "%d", &debug);
			break;
		case 'i':
			sscanf(optarg, "%s", input_raw);
			break;
		case 's':
			sscanf(optarg, "%s", output_sequential);
			break;
		case 'p':
			sscanf(optarg, "%s", output_simd);
			break;
		case '?':
		if (optopt == 'i' || optopt == 's' || optopt == 'p' || optopt == 'N' || optopt == 'D')
				printf("Option -%c required.\n", optopt);
		else if (isprint (optopt))
				printf("Option unknown `-%c'.\n", optopt);
		else
				printf("Unknown`\\x%x'.\n",optopt);
		return 1;
		default:
		abort();
		}
    /*Validate arguments*/
	FILE* input;
	FILE* sequential;
	FILE* simd;
	if(size < 0) {
		printf("Size %d must be positive\n", size);
		return 1;
	}
	else if( !(debug == 1 || debug == 0)) {
		printf("Debug value %d must be 0 or 1\n", debug);
		return 1;
	}
	else if( !(input = fopen(input_raw, "rb")) ){
		printf("File %s not found\n", input_raw);
		return 1;
	}
	else if( !(sequential = fopen(output_sequential, "wb")) ){
		printf("File %s cannot be created\n", output_sequential);
		return 1;
	}
	else if( !(simd = fopen(output_simd, "wb")) ){
		printf("File %s cannot be created\n", output_simd);
		return 1;
	}
	image *img = readImage(input, size);

	if(debug){
		printf("\nOriginal image\n");
		printImage(img);
		printf("\n");
	}

	image *result = blankImage(size, img);
	dilation_seq(img, result);
	
	if(debug){
		printf("Sequential image dilated\n");
		printImage(result);
		printf("\n");
	}

	
	image *result2 = blankImage(size, img);
	dilation_simd(img, result2);

	if(debug){
		printf("SIMD image dilation\n");
		printImage(result2);
		printf("\n");
	}


	fprintImage(result, sequential);
	fprintImage(result2, simd);
	return 0;
}