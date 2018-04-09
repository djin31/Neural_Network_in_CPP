#include "neuralnetwork.cpp"
#include <fstream>

data * traindata;
data * testdata;

using namespace std;

void readdata()
{	
	ifstream file;
    int number_of_test_images,number_of_train_images;

    file.open("traindata.txt", ios::in);
    if (!file)
    {
     cout<<"File not found\n";   
    }
    file>>number_of_train_images;
    traindata = new data[number_of_train_images];
	for (int i=0; i<number_of_train_images; i++)
	{
		for(int j=0; j<IMG_SIZE; j++)
		{
			file>>traindata[i].x[j];
		}
	}
	for (int i=0; i<number_of_train_images; i++)
	{
		for(int j=0; j<OUT_SIZE; j++)
		{
			file>>traindata[i].y[j];
		}
	}
	file.close();

	file.open("testdata.txt", ios::in);
    if (!file)
    {
     cout<<"File not found\n";   
    }
    file>>number_of_test_images;
    testdata = new data[number_of_test_images];
	for (int i=0; i<number_of_test_images; i++)
	{
		for(int j=0; j<IMG_SIZE; j++)
		{
			file>>testdata[i].x[j];
		}
	}
	for (int i=0; i<number_of_test_images; i++)
	{
		for(int j=0; j<OUT_SIZE; j++)
		{
			file>>testdata[i].y[j];
		}
	}
	
	file.close();

}
int main(int argc, char const *argv[])
{
	int *sizes = new int[1];
	sizes[0]=32;
	cout<<"Loading data\n";
	double t = omp_get_wtime();
	readdata(); 
	cout<<"Loaded data in "<<omp_get_wtime()-t<<endl;
	cout<<"Configuring neural network\n";
	neuralnetwork mynetwork = neuralnetwork(1,sizes,5000,traindata,1000,testdata,32);
	cout<<"Configuring sgd\n";
	mynetwork.SGD(10,500,0.01);
	return 0;
}