#include <time.h>
#include<stdlib.h>
#include<iostream>
#include <omp.h>

#define size 100
#define size2 784
#define size3 10000
#define blocksize 64

double **a;
double **b;
double **c;
double **d;


int main(int argc, char const *argv[])
{
	a = new double*[size];
	for (int i = 0; i<size;i++)
		a[i]=new double[size2];

		b = new double*[size2];
	for (int i = 0; i<size2;i++)
		b[i]=new double[size3];

		c = new double*[size];
	for (int i = 0; i<size;i++)
		c[i]=new double[size3];

		d = new double*[size];
	for (int i = 0; i<size;i++)
		d[i]=new double[size3];
	srand(time(NULL));
	for (int i=0;i<size;i++)
		for(int j=0;j<size2;j++)
			a[i][j]=1;
	for (int i=0;i<size2;i++)
		for(int j=0;j<size3;j++)
			b[i][j]=1;
	
	double t = omp_get_wtime();
	for (int i=0;i<size;i++)
		for(int j=0;j<size3;j++)
		{
			double t=0;
			for (int k=0;k<size2;k++)
				t+=a[i][k]*b[k][j];
			c[i][j]=t;
		}
	t=omp_get_wtime()-t;
	std::cout<<"time "<<t<<"\n";


	int i,j,ii,jj,iimax,jjmax,k,kk,kkmax;
	double tmp;
	t = omp_get_wtime();

	#pragma omp parallel for collapse(2)
	for(int i=0;i<size;i++)
		for(int j=0;j<size3;j++)
			d[i][j]=0;
	#pragma omp parallel for collapse(2) schedule (static,64) private(i,j,ii,jj,iimax,jjmax,k,kkmax,kk) 
		for (i=0;i<size;i+=blocksize)
		{	
			
			for( j=0;j<size3;j+=blocksize)
			{
				iimax=std::min(i+blocksize,size);			
				jjmax= std::min(j+blocksize,size3);

			
				for(k=0;k<size2;k+=blocksize)
				{
					kkmax = std::min(k+blocksize,size2);
					for (ii=i;ii<iimax;ii++)
					{
						
						for (jj=j;jj<jjmax;jj++)
							for (kk=k;kk<kkmax;kk++)
							d[ii][jj]+=a[ii][kk]*b[kk][jj];
					}
				}

				
			}
		}

	t=omp_get_wtime()-t;
	std::cout<<"time "<<t<<"\n";
							

	for (int i=0;i<size;i++)
		for(int j=0;j<size3;j++)
		{

			if (abs(d[i][j]-c[i][j])>0.01)
				{std::cout<<"\ndhokha\n";std::cout<<d[i][j]<<" "<<c[i][j];exit(0);}
		}

	/*	for (int i=0;i<size;i++)
	{
		for(int j=0;j<size2;j++)
			std::cout<<a[i][j]<<" ";
		std::cout<<"\n";
	}
	for (int i=0;i<size2;i++)
	{
		for(int j=0;j<size3;j++)
			std::cout<<b[i][j]<<" ";
		std::cout<<"\n";
	}
	for (int i=0;i<size;i++)
	{
		for(int j=0;j<size3;j++)
			std::cout<<c[i][j]<<" ";
		std::cout<<"\n";
	}
	for (int i=0;i<size;i++)
	{
		for(int j=0;j<size3;j++)
			std::cout<<c[i][j]<<" ";
		std::cout<<"\n";
	}*/

	std::cout<<"pyaar\n";
	//std::getchar();
	return 0;
}
