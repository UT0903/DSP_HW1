#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "hmm.h"
#define SEQ_NUM 2500 
#define SEQ_LEN 50
#define STATE_NUM 6
#define MODEL_NUM 5
unsigned char seq[SEQ_NUM][SEQ_LEN];
HMM model[MODEL_NUM];
double delta[SEQ_NUM][MODEL_NUM][SEQ_LEN][STATE_NUM];

void loadSeq(const char *seq_path){
	int tmp_idx = 0;
	char tmp_seq[SEQ_LEN];
	FILE *fp = fopen(seq_path, "rt");
	while(fscanf(fp, "%s", tmp_seq) != EOF){
		for(int i = 0; i < (int)strlen(tmp_seq); i++){
			seq[tmp_idx][i] = tmp_seq[i] - 'A';
		}
		tmp_idx++;
	}
	//printf("Load finish!\n");
	/*for(int i = 0; i < SEQ_LEN; i++){
		printf("%d", data->seq[0][i]);
	}*/
	return;
}

double Max(double i, double j){
	return ((i > j)? i:j);
}
void Viterbi(FILE *fp, FILE *fr){
	char model_name[MODEL_NUM][50];
	int e;
	for(int i = 0; i < MODEL_NUM; i++){
		e = fscanf(fr, "%s", model_name[i]);
	}
	for(int sn = 0; sn < SEQ_NUM; sn++){
		double max[MODEL_NUM];
		for(int mn = 0; mn < MODEL_NUM; mn++){
			//Init
			for(int i = 0; i < STATE_NUM; i++){
				delta[sn][mn][0][i] = model[mn].initial[i] * model[mn].observation[seq[sn][0]][i];
			}
			//Recur
			for(int t = 1; t < SEQ_LEN; t++){
				for(int j = 0; j < STATE_NUM; j++){
					double temp_max = 0.0;
					for(int i = 0; i < STATE_NUM; i++){
						temp_max = Max(temp_max, delta[sn][mn][t-1][i] * model[mn].transition[i][j]);
					}
					delta[sn][mn][t][j] = temp_max * model[mn].observation[seq[sn][t]][j];
				}
			}
			//Termination
			max[mn] = -1.0;
			for(int i = 0; i < STATE_NUM; i++){
				max[mn] = Max(max[mn], delta[sn][mn][SEQ_LEN-1][i]);
			}
		}
		int best_idx = 0;
		double best = max[0];
		for(int mn = 1; mn < MODEL_NUM; mn++){
			if(max[mn] > max[best_idx]){
				best_idx = mn;
				best = max[mn];
			}
		}
		fprintf(fp, "%s %e\n", model_name[best_idx], best);
	}
	fclose(fp);
	return;
}

void Evaluation(FILE *fr, FILE *fa){
	char result[50], e[50], answer[50];
	int yes = 0, total = 0;
	int g;
	while(fscanf(fr, "%s %s", result, e) != EOF){
		g = fscanf(fa, "%s", answer);
		if(strcmp(result, answer) == 0){
			yes++;
		}
		total++;
	}
	printf("data size: %d, accepted: %d, accur = %lf\n", total, yes, (double)yes / (double)total);
}




int main(int argc, char *argv[]){
#ifdef EVAL
	if(argc != 5){
		fprintf(stderr, "Evaluation mode usage: ./test <models_list_path> <seq_path> <output_result_path> <answer_file_path>\n");
		exit(-1);
	}
#else
	if(argc != 4){
		fprintf(stderr, "usage: ./test <models_list_path> <seq_path> <output_result_path>\n");
		exit(-1);
	}
#endif
	char models_list_path[50], seq_path[50], output_result_path[50], answer_file_path[50];
	strcpy(models_list_path, argv[1]);
	strcpy(seq_path, argv[2]);
	strcpy(output_result_path, argv[3]);
#ifdef EVAL
	strcpy(answer_file_path, argv[4]);
#endif	

	assert(MODEL_NUM == load_models(models_list_path, model, MODEL_NUM));
	loadSeq(seq_path);
	
	Viterbi(open_or_die(output_result_path, "w+"), open_or_die(models_list_path, "r"));
#ifdef EVAL
	Evaluation(open_or_die(output_result_path, "r"), open_or_die(answer_file_path, "r"));
#endif

}