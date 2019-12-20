# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <ctype.h>
# include <unistd.h>
# include <sys/time.h>

__host__ int main(int argc, char const *argv[]) {
	int grid_size, block_size_x, block_size_y, steps, t;
	/*Parameters of the program: 
	-N    grid size
	-x    block size x
	-y    block size y
	-T    number of steps
	-f    output file
	-t    output iteration
	*/
	opterr = 0;
	printf("Parameters %i\n", argc);
	int c;
	char* f = (char*) malloc(sizeof(char) * 50);
	
	while ((c = getopt (argc, argv, "N:x:y:T:f:t:")) != -1)
		switch (c) {
			case 'N':
				sscanf(optarg, "%d", &i);
				break;
			case 'x':
				a = atof(optarg);
				break;
			case 'y':
				b = atof(optarg);
				break;
			case 'T':
				c = atof(optarg);
				break;
			case 'f':
				sscanf(optarg, "%s", f);
				break;
			case 't':
				s = atof(optarg);
				break;
			case '?':
				if (optopt == 'i' || optopt == 'a' || optopt == 'b' || optopt == 'c' || optopt == 'd' || optopt == 's' || optopt == 'f')
						printf("Option -%c required.\n", optopt);
				else if (isprint (optopt))
						printf("Option unknown `-%c'.\n", optopt);
				else
						printf("Unknown`\\x%x'.\n", optopt);
				return 1;
			default:
				abort();
			}
	//printf("- i: %i\n- a: %f\n- b: %f\n- c: %f\n- d: %f\n- s: %f\n- f: %s\n", i, a, b, c, d, s, f);
	return 0;
}