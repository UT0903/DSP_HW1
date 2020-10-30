#include <stdio.h>
#include "hmm.h"
#define SEQ_LEN 50
#define SEQ_NUM 10000
#define STATE_NUM 6
#define OBSERVE_TYPE 6
typedef struct{
	unsigned char seq[SEQ_NUM][SEQ_LEN];
	double alpha[SEQ_NUM][SEQ_LEN][STATE_NUM];
	double beta[SEQ_NUM][SEQ_LEN][STATE_NUM];
	double gamma[SEQ_NUM][SEQ_LEN][STATE_NUM];
	double epsilon[SEQ_NUM][SEQ_LEN][STATE_NUM][STATE_NUM];
}Data;


void loadSeq(Data *data, const char *seq_path){
	int tmp_idx = 0;
	char tmp_seq[SEQ_LEN];
	FILE *fp;
	fp = fopen(seq_path, "rt");
	while(fscanf(fp, "%s", tmp_seq) != EOF){
		for(int i = 0; i < (int)strlen(tmp_seq); i++){
			data->seq[tmp_idx][i] = tmp_seq[i] - 'A';
		}
		tmp_idx++;
	}
	//printf("Load finish!\n");
	/*for(int i = 0; i < SEQ_LEN; i++){
		printf("%d", data->seq[0][i]);
	}*/
	return;
}


void compute_alpha(HMM *model, Data *data){
#pragma omp parallel for
	for(int sn = 0; sn < SEQ_NUM; sn++){
		//Init
//		printf("print alpha:\n");
		for(int i = 0; i < STATE_NUM; i++){
			data->alpha[sn][0][i] = model->initial[i] * model->observation[data->seq[sn][0]][i];
//			printf("%lf ", data->alpha[sn][0][i]);
		}
		for(int t = 0; t < SEQ_LEN - 1; t++){
			for(int j = 0; j < STATE_NUM; j++){
				double sum = 0.0;
				for(int i = 0; i < STATE_NUM; i++){
					sum += data->alpha[sn][t][i] * model->transition[i][j];
				}
				data->alpha[sn][t+1][j] = sum * model->observation[ data->seq[sn][t+1] ][j];
//				printf("%lf ", data->alpha[sn][t+1][j]);
			}
//			printf("\n");
		}
	}
	/*for(int i = 0; i < STATE_NUM; i++){
		printf("%d", );
	}*/
}

void compute_beta(HMM *model, Data *data){
#pragma omp parallel for
	for(int sn = 0; sn < SEQ_NUM; sn++){
		//Init
//		printf("print beta:\n");
		for(int i = 0; i < STATE_NUM; i++){
			data->beta[sn][SEQ_LEN - 1][i] = 1;
//			printf("%.64lf ", data->beta[sn][SEQ_LEN - 1][i]);
		}
//		printf("\n");
		for(int t = SEQ_LEN - 2; t >= 0; t--){
			for(int i = 0; i < STATE_NUM; i++){
				double sum = 0.0;
				for(int j = 0; j < STATE_NUM; j++){
					sum += model->transition[i][j] * \
					model->observation[ data->seq[sn][t+1] ][j] * data->beta[sn][t+1][j];
				}
				data->beta[sn][t][i] = sum;
//				printf("%.64lf ", data->beta[sn][t][i]);
			}
//			printf("\n");
		}
	}
	return;
}

void compute_gamma(Data *data){
#pragma omp parallel for
	for(int sn = 0; sn < SEQ_NUM; sn++){
//		printf("print gamma:\n");
		for(int t = 0; t < SEQ_LEN; t++){
			double sum = 0.0;
			for(int i = 0; i < STATE_NUM; i++){
//				printf("test:\n%.90lf\n%.90lf\n%.90lf\n", data->alpha[sn][t][i], data->beta[sn][t][i], sum);
				sum += data->alpha[sn][t][i] * data->beta[sn][t][i];
			}
			//printf("sum = %.90lf\n", sum);
			for(int i = 0; i < STATE_NUM; i++){
				data->gamma[sn][t][i] = data->alpha[sn][t][i] * data->beta[sn][t][i] / sum;
//				printf("t = %d, %lf ", t, data->gamma[sn][t][i]);
			}
//			printf("\n");
		}
	}
	return;
}

void compute_epsilon(HMM *model, Data *data){
#pragma omp parallel for 
	for(int sn = 0; sn < SEQ_NUM; sn++){
		for(int t = 0; t < SEQ_LEN - 1; t++){
			double sum = 0.0;
			for(int i = 0; i < STATE_NUM; i++){
				for(int j = 0; j < STATE_NUM; j++){
					sum += data->alpha[sn][t][i] * model->transition[i][j] * \
					model->observation[ data->seq[sn][t+1] ][j] * \
					data->beta[sn][t+1][j];
				}
			}
			for(int i = 0; i < STATE_NUM; i++){
				for(int j = 0; j < STATE_NUM; j++){
					data->epsilon[sn][t][i][j] = data->alpha[sn][t][i] * \
					model->transition[i][j] * \
					model->observation[ data->seq[sn][t+1] ][j] * \
					data->beta[sn][t+1][j] / sum;
				}
			}
		}
	}
	return;
}
void renew_model_a(HMM *model, Data *data){
#pragma omp parallel for
	for(int i = 0; i < STATE_NUM; i++){	
		double upper_sum[STATE_NUM], lower_sum = 0.0;
		for(int j = 0; j < STATE_NUM; j++){
			double upper = 0.0;
			for(int sn = 0; sn < SEQ_NUM; sn++){
				for(int t = 0; t < SEQ_LEN - 1; t++){
					upper += data->epsilon[sn][t][i][j];
				}
			}
			upper_sum[j] = upper;
		}
		for(int sn = 0; sn < SEQ_NUM; sn++){
			for(int t = 0; t < SEQ_LEN - 1; t++){
				lower_sum += data->gamma[sn][t][i];
			}
		}
		for(int j = 0; j < STATE_NUM; j++){
			model->transition[i][j] = upper_sum[j] / lower_sum;
		}
	}
}
void renew_model_b(HMM *model, Data *data){
#pragma omp parallel for collapse(2)
	for(int i = 0; i < STATE_NUM; i++){
		for(int k = 0; k < OBSERVE_TYPE; k++){
			double upper = 0.0;
			for(int sn = 0; sn < SEQ_NUM; sn++){
				for(int t = 0; t < SEQ_LEN; t++){
					if(data->seq[sn][t] == k){
						upper += data->gamma[sn][t][i];
					}
				}
			}
			double lower = 0.0;
			for(int sn = 0; sn < SEQ_NUM; sn++){
				for(int t = 0; t < SEQ_LEN; t++){
					lower += data->gamma[sn][t][i];
				}
			}
			model->observation[k][i] = upper / lower;
		}
	}
	return;
}
void renew_model_pi(HMM *model, Data *data){
#pragma omp parallel for
	for(int i = 0; i < STATE_NUM; i++){
		double sum = 0.0;
		for(int sn = 0; sn < SEQ_NUM; sn++){
			sum += data->gamma[sn][0][i];
		}
		model->initial[i] = sum / SEQ_NUM;
	}
	return;
}