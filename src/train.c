#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hmm.h"
#include "compute.h"
HMM model;
Data data;
int main(int argc, char *argv[]){
	if(argc != 5){
		fprintf(stderr, "usage: ./train <iter> <model_init_path> <seq_path> <output_model_path>\n");
		exit(-1);
	}
	int iter = atoi(argv[1]);
	char model_init_path[50], seq_path[50], output_model_path[50];
	strcpy(model_init_path, argv[2]);
	strcpy(seq_path, argv[3]);
	strcpy(output_model_path, argv[4]);
	
	loadHMM(&model, model_init_path);
	
	loadSeq(&data, seq_path);

	for(int i = 0; i < iter; i++){
		compute_alpha(&model, &data);
		compute_beta(&model, &data);
		compute_gamma(&data);
		compute_epsilon(&model, &data);
		renew_model_a(&model, &data);
		renew_model_b(&model, &data);
		renew_model_pi(&model, &data);
	}
	dumpHMM(open_or_die(output_model_path, "w+"), &model);
}