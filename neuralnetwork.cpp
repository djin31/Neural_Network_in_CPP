#define IMG_SIZE 784
#define OUT_SIZE 10
#define MAX_LAYERS 10

#include <iostream>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>

using namespace std;

//global declarations
typedef struct DATA
{
	double x[IMG_SIZE];
	double y[OUT_SIZE];

} data;

typedef struct NETWORK
{
	int num_layers;
	int sizes[MAX_LAYERS];
	double **biases;   /* array of bias vectors (one for each layer) */
	double ***weights; /* array of 2d weight matrices (for each layer) */
	double **delta_biases;
	double ***delta_weights;
	double ***derivatives;

} network;

class neuralnetwork
{
  private:
	data *traindata;
	data *testdata;
	network *net;

	int train_set_size;
	int test_set_size;
	int blocksize;

	double ***train_activations;
	double **train_results;
	double ***test_activations;

	int epochs;
	int batch_size;
	int learning_rate;

	vector<int> randomising_array;

  public:
	// utility functions
	double sigmoid(double z)
	{
		return (1.0 / (1.0 + exp(-z)));
	}

	double sigmoid_prime(double z)
	{
		return (sigmoid(z) * (1 - sigmoid(z)));
	}

	/* Matrix multiply to compute: C = A x B.
	* A is mxn, B is nxp, C is mxp
	*/
	void MatMul(int m, int n, int p, int b, double **A, double **B, double **C)
	{
		int i, j, ii, jj, iimax, jjmax, k, kk, kkmax;
#pragma omp parallel for collapse(2)
		for (int i = 0; i < m; i++)
			for (int j = 0; j < p; j++)
				C[i][j] = 0;
#pragma omp parallel for collapse(2) schedule(static, 64) private(i, j, k, ii, jj, kk, iimax, jjmax, kkmax)
		for (i = 0; i < m; i += b)
		{
			for (j = 0; j < p; j += b)
			{
				iimax = std::min(i + b, m);
				jjmax = std::min(j + b, p);

				for (k = 0; k < n; k += b)
				{
					kkmax = std::min(k + b, n);
					for (ii = i; ii < iimax; ii++)
					{

						for (jj = j; jj < jjmax; jj++)
							for (kk = k; kk < kkmax; kk++)
								C[ii][jj] += A[ii][kk] * B[kk][jj];
					}
				}
			}
		}
	}

	//Constructor configures neural network
	neuralnetwork(int num_layers, int *sizes, int number_of_train_imgs, data *train_data, int number_of_test_imgs, data *test_data, int bsize)
	{
		srand(time(NULL));
		net = new network();
		net->num_layers = num_layers;
		blocksize = bsize;

		net->sizes[0] = IMG_SIZE;
		for (int i = 0; i < num_layers; i++)
		{
			net->sizes[i + 1] = sizes[i];
		}
		net->sizes[num_layers + 1] = OUT_SIZE;

		//Initialising biases, the last biases are for the output layer
		cout << "Initialising biases\n";
		net->biases = (double **)malloc((num_layers + 1) * sizeof(double *));
		for (int i = 0; i <= num_layers; i++)
		{
			net->biases[i] = (double *)malloc(net->sizes[i + 1] * sizeof(double));

			for (int j = 0; j < net->sizes[i + 1]; j++)
			{
				net->biases[i][j] = (double)rand() / RAND_MAX;
			}
		}

		net->delta_biases = (double **)malloc((num_layers + 1) * sizeof(double *));
		for (int i = 0; i <= num_layers; i++)
		{
			net->delta_biases[i] = (double *)malloc(net->sizes[i + 1] * sizeof(double));
		}

		//Initialising weights, the first weights are for the input layer
		cout << "Initialising weights\n";
		net->weights = (double ***)malloc((num_layers + 1) * sizeof(double **));
		//first index identifies layer
		//second index identifies neuron in the next activation layer
		//third identifies the neuron in current activation layer

		for (int i = 0; i <= num_layers; i++)
		{
			net->weights[i] = (double **)malloc(net->sizes[i + 1] * sizeof(double *));
			for (int j = 0; j < net->sizes[i + 1]; j++)
			{
				net->weights[i][j] = (double *)malloc(net->sizes[i] * sizeof(double));
				for (int k = 0; k < net->sizes[i]; k++)
				{
					net->weights[i][j][k] = (double)rand() / RAND_MAX;
					//cout<<net->weights[i][j][k] <<" ";
				}
				//cout<<endl;
			}
		}

		net->delta_weights = (double ***)malloc((num_layers + 1) * sizeof(double **));
		for (int i = 0; i <= num_layers; i++)
		{
			net->delta_weights[i] = (double **)malloc(net->sizes[i + 1] * sizeof(double *));

			for (int j = 0; j < net->sizes[i + 1]; j++)
			{
				net->delta_weights[i][j] = (double *)malloc(net->sizes[i] * sizeof(double));
			}
		}

		train_set_size = number_of_train_imgs;
		traindata = (data *)malloc(train_set_size * sizeof(data));
		traindata = train_data;
		test_set_size = number_of_test_imgs;
		testdata = (data *)malloc(test_set_size * sizeof(data));
		testdata = test_data;

		for (int i = 0; i < train_set_size; i++)
			randomising_array.push_back(i);

		//initialising test activations
		cout << "Initialising test_activations\n";
		test_activations = new double **[num_layers + 2];
		for (int i = 0; i <= (num_layers + 1); i++)
		{
			test_activations[i] = (double **)malloc(net->sizes[i] * sizeof(double *));
			for (int j = 0; j < net->sizes[i]; j++)
			{
				test_activations[i][j] = (double *)malloc((number_of_test_imgs) * sizeof(double));
			}
		}

		for (int i = 0; i < number_of_test_imgs; i++)
		{
			for (int j = 0; j < IMG_SIZE; j++)
			{
				test_activations[0][j][i] = testdata[i].x[j];
			}
		}

	}

	void feedforward(double ***activations, int number_of_imgs)
	{
		for (int i = 0; i <= (net->num_layers); i++)
		{
			MatMul(net->sizes[i + 1], net->sizes[i], number_of_imgs, blocksize, net->weights[i], activations[i], activations[i + 1]);

			for (int j = 0; j < net->sizes[i + 1]; j++)
			{
				for (int k = 0; k < number_of_imgs; k++)
					activations[i + 1][j][k] = sigmoid(activations[i + 1][j][k] + net->biases[i][j]);
			}
		}
	}

	void cost_derivative()
	{
		int num_layers = net->num_layers;
		double temp;
		for (int p = 0; p < OUT_SIZE; p++)
		{
			for (int i = 0; i < batch_size; i++)
			{
				net->derivatives[num_layers][p][i] = 2 * (train_activations[num_layers + 1][p][i] - train_results[p][i]) * ((train_activations[num_layers + 1][p][i]) * (1 - train_activations[num_layers + 1][p][i]));
			}
		}
		for (int n = (num_layers - 1); n >= 0; n--)
		{
			for (int p = 0; p < (net->sizes[n + 1]); p++)
			{
				for (int i = 0; i < batch_size; i++)
				{
					temp = 0;
					for (int k = 0; k < (net->sizes[n + 2]); k++)
						temp += (net->derivatives[n + 1][k][i] * net->weights[n + 1][k][p]);
					net->derivatives[n][p][i] = temp * (train_activations[n+1][p][i]) * (1 - train_activations[n +1][p][i]);
				}
			}
		}
		/*for (int i=0;i<10;i++)
		{
			cout<<net->derivatives[0][i][0]<<endl;
		}*/

		
		for (int n = 0; n < (num_layers + 1); n++)
		{
			for (int p = 0; p < (net->sizes[n + 1]); p++)
			{
				net->delta_biases[n][p] = 0;
				for (int i = 0; i < batch_size; i++)
				{
					net->delta_biases[n][p] += net->derivatives[n][p][i];
				}
			}
		}
		//cout<<net->delta_biases[1][5];

		
		for (int n = 0; n < (num_layers + 1); n++)
		{
			for (int p = 0; p < (net->sizes[n + 1]); p++)
			{
				for (int q = 0; q < net->sizes[n]; q++)
				{
					net->delta_weights[n][p][q] = 0;
					for (int i = 0; i < batch_size; i++)
					{
						net->delta_weights[n][p][q] += net->derivatives[n][p][i] * train_activations[n][q][i];
					}
				}
			}
		}
	}

	void update_mini_batch()
	{
		int num_layers = net->num_layers;

		//compute delta_biases and delta_weights
		cost_derivative();
		//updating weights and biases
		for (int i = 0; i <= num_layers; i++)
		{
			for (int j = 0; j < net->sizes[i + 1]; j++)
			{
				net->biases[i][j] -= learning_rate * net->delta_biases[i][j] / batch_size;
			}
		}

		for (int i = 0; i <= num_layers; i++)
		{
			for (int j = 0; j < net->sizes[i + 1]; j++)
			{
				for (int k = 0; k < net->sizes[i]; k++)
				{
					net->weights[i][j][k] -= learning_rate * net->delta_weights[i][j][k] / batch_size;
				}
			}
		}
		for (int i =0 ;i<10;i++)
		{
			cout<<net->weights[1][0][i]<<" ";
		}
		cout<<endl;
	}

	int evaluate()
	{
		feedforward(test_activations, test_set_size);

		int c = 0;
		double m;
		int mv, tv;
		for (int i = 0; i < test_set_size; i++)
		{
			m = 0;
			for (int j = 0; j < OUT_SIZE; j++)
			{
				if (m < test_activations[net->num_layers + 1][j][i])
				{
					m = test_activations[net->num_layers + 1][j][i];
					mv = j;
				}

				if (fabs(testdata[i].y[j] - 1) < 0.01)
					tv = j;
				//cout<<test_activations[net->num_layers+1][j][i]<<" ";
			}

			if (tv == mv)
			{
				c++;
				//cout << tv << endl;
			}
		}
		
		return c;
	}

	//configures stochastic gradient descent parameters
	void SGD(int epochs, int mini_batch_size, double eta)
	{
		this->epochs = epochs;
		batch_size = mini_batch_size;
		learning_rate = eta;
		srand(time(NULL));

		int c;
		double d;
		//initialising training activations
		cout << "Allocating training activations\n";
		train_activations = (double ***)malloc((net->num_layers + 2) * sizeof(double **));
		for (int i = 0; i <= (net->num_layers + 1); i++)
		{
			train_activations[i] = (double **)malloc(net->sizes[i] * sizeof(double *));
			for (int j = 0; j < net->sizes[i]; j++)
			{
				train_activations[i][j] = (double *)malloc((batch_size) * sizeof(double));
			}
		}

		train_results = (double **)malloc(OUT_SIZE * sizeof(double *));
		for (int i = 0; i < OUT_SIZE; i++)
		{
			train_results[i] = (double *)malloc(batch_size * sizeof(double));
		}

		net->derivatives = (double ***)malloc((net->num_layers + 1) * sizeof(double **));
		for (int i = 0; i <= (net->num_layers); i++)
		{
			net->derivatives[i] = (double **)malloc(net->sizes[i + 1] * sizeof(double *));
			for (int j = 0; j < net->sizes[i + 1]; j++)
			{
				net->derivatives[i][j] = (double *)malloc(batch_size * sizeof(double));
			}
		}

		cout << "Beginning epochs\n";
		for (int i = 0; i < epochs; i++)
		{
			random_shuffle(randomising_array.begin(), randomising_array.end());
			int r;

			for (int j = 0; j < train_set_size; j += batch_size)
			{
				for (int k = 0; k < batch_size; k++)
				{
					r = randomising_array.at(j + k);
					for (int p = 0; p < IMG_SIZE; p++)
					{
						train_activations[0][p][k] = traindata[r].x[p];
					}
					for (int p = 0; p < OUT_SIZE; p++)
					{
						train_results[p][k] = traindata[r].y[p];
					}
				}
				feedforward(train_activations, batch_size);
				update_mini_batch();
			}
			c = evaluate();
			d = ((double)c) / test_set_size;
			cout << "Epoch " << i << ": Accuracy " << c << endl;
		}
	}
};
