#include "neuralnetwork_info_spitter.cpp"
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
	sizes[0]=15;
	readdata(); 	
	neuralnetwork mynetwork = neuralnetwork(1,sizes,50000,traindata,100,testdata,512);
	mynetwork.SGD(5,1,1);
	mynetwork.SGD(5,4,1);
	mynetwork.SGD(5,16,1);
	mynetwork.SGD(5,64,1);
	mynetwork.SGD(5,256,1);
	return 0;
}
