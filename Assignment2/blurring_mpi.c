#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// gcc bluring_mpi.c -lm
// ./a.out file.pgm N kernel_name


float* kernel(int name, float f, int N);

u_int8_t* bluring(float* K, u_int8_t* im, int N, int h, int w);

int main(int argc ,char **argv){

	if(argc<5){
		printf("You must provide three arguments in executions:\n file_name.pgm,  kernel dimension, kernel case number (1 for mean, 2 for weight, 3 for gaussian ), and 'f' for the weighted kernel\n");
		exit(1);
	}

	//###### LETTURA ######################################################
	char* filename = argv[1];
	FILE* fo=fopen(filename,"rb");
    if (!fo) {
        printf("Error: Unable to open file %s.\n\n", filename);
        exit(1);
    }
    printf("opening %s\n",filename);

    char type[2];
	int width=0,height=0,maxval=0;
	fscanf(fo,"%2s %d %d %d",type,&width,&height,&maxval);
	printf("HEADER:  type=%s, width=%d, height=%d, maxval=%d\n",type,width,height, maxval);

	u_int8_t* im=(u_int8_t *)malloc(height*width*sizeof(u_int8_t));
	fread(im,sizeof(u_int8_t), width*height,fo);     
	fclose(fo);
	/*
	FILE* ff=fopen("prova.pgm","wb");
	fprintf(ff,"%2s\n%d %d\n%d\n", type, width, height, maxval);
	fwrite(im,sizeof(u_int8_t), width*height, ff);
    fclose(ff);
	*/

	//###### BLURING ######################################################
	int N=strtol(argv[2], NULL, 10);
	if(N<=0 || N%2==0){
		printf("The dimension of the kernel should be a positive and odd integer.\n");
		exit(1);
	}
	int name=strtol(argv[3],NULL,10);
	float f=strtof(argv[4],NULL);
	if(f<0 || f>1){
		printf("'f' must be in the interval [0,1]. \n");
		exit(1);
	}
	float* K= kernel(name, f, N);
	
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			printf("%f ",K[i*N+j]);
		}
		printf("\n");
	}
	
	

	u_int8_t* blur=bluring(K,im,N,height,width);


	/*
	char fn[16];
	snprintf(fn, sizeof(fn), "blur_mean_%d.pgm", N);
	printf("Output: %s \n",fn);
	FILE* ff=fopen(fn,"wb");*/
    
	FILE* ff=fopen("blur_mean_9.pgm","wb");
	fprintf(ff,"%2s\n%d %d\n%d\n", type, width, height, maxval);
	fwrite(blur,sizeof(u_int8_t), width*height, ff);
    fclose(ff);
 
	/*
	int myid , numprocs , proc ;
	
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	if ( argc <=1) { //Security exit
		fprintf (stderr , " Usage : mpi -np n %s number_of_iterations \n", argv[0] ) ;
		MPI_Finalize() ;
		exit(-1) ;
	}
	*/
	
	free(im);
	free(K);
	free(blur);
	return 0;
}


float * kernel(int name,float f, int N){
	
	float* k=(float *)malloc(N*N*sizeof(float));
	int n=N/2;
	
	switch(name){
		case(1): // MEAN (non normalizzato)
			for (int i=0; i<N; i++){
				for (int j=0; j<N; j++){
					k[i*N+j]=1;
				}
			}
			return k;
		case(2): // WEIGHT (normalizzato)
			for (int i=0; i<N; i++){
				for (int j=0; j<N; j++){
					k[i*N+j]=(1-f)/((float)N*(float)N-1);
				}
			}
			k[n*(N+1)]=f;
			return k;
		case(3): // GAUSSIAN (non normalizzato)
			for (int i=0; i<N; i++){
				for (int j=0; j<N; j++){
					k[i*N+j]=pow(2.7182,(-(pow(i-n,2)+pow(j-n,2))/(2*n*n)))/(2*3.1415*n*n);
				}
			}
			return k;
	}
}

			
u_int8_t* bluring(float* K, u_int8_t* im, int N, int h, int w){
	int n=N/2;
	float norm, sum;
	u_int8_t* blur=(u_int8_t *)malloc(h*w*sizeof(u_int8_t));
	/*
	for (int i=N-(n+1); i<h-(N-n); i++){
		for (int j=N-(n+1); j<w-(N-n); j++){
			sum=0;
			for (int u=-n; u<=n; u++){
				for (int v=-n; v<=n; v++){
					sum += im[(i+u)*h+(j+v)]*K[(u+n)*N+(v+n)];
					
				}
			}
			blur[i*h+j] = (u_int8_t)sum;
		}
	}
	*/
	int e,f,g,l;
	for (int i=0; i<h; i++){
		for (int j=0; j<w; j++){
			sum=0;
			if( i<N-(n+1) ){
				e=-i;
				f=n;
				if( j<N-(n+1) ){
					g=-j;
					l=n;
				}
				else if( j>=w-(N-n) ){
					g=-n;
					l=w-j-1;
				}
				else{
					g=-n;
					l=n;
				}
			}
			else if( i>=h-(N-n) ){
				e=-n;
				f=h-i-1;
				if( j<N-(n+1) ){
					g=-j;
					l=n;
				}
				else if( j>=w-(N-n) ){
					g=-n;
					l=w-j-1;
				}
				else{
					g=-n;
					l=n;
				}
			}
			else{
				e=-n;
				f=n;
				if( j<N-(n+1) ){
					g=-j;
					l=n;
				}
				else if( j>=w-(N-n) ){
					g=-n;
					l=w-j-1;
				}
				else{
					g=-n;
					l=n;
				}	
			}
			/*
			e=-n;
			f=n;
			g=-n;
			l=n;
			*/
			for (int u=e; u<=f; u++){
				for (int v=g; v<=l; v++){
					sum += K[(u+n)*N+(v+n)];
				}
			}
			norm=1./sum;
			sum=0;
			for (int u=e; u<=f; u++){
				for (int v=g; v<=l; v++){
					sum += im[(i+u)*h+(j+v)]*K[(u+n)*N+(v+n)]*norm;
				}
			}
			blur[i*h+j] = (u_int8_t)sum;
		}
	}

/*	
	for (int i=0; i<N-(n+1); i++){
		for (int j=0; j<N-(n+1); j++){
			sum=0;
			for (int ki=n-i; ki<N; ki++ ){
				for (int kj=n-j; kj<N; kj++){
					sum += K[ki*N+kj];
				}
			}
			norm=1.0/sum;
			sum=0;
			for (int u=-i; u<=n; u++){
				for (int v=-j; v<=n; v++){
					sum += im[(i+u)*h+(j+v)]*K[(u+n)*N+(v+n)]*norm;
					
				}
			}
			blur[i*h+j] = (u_int8_t)sum;
		}
	}

	for (int i=h-(N-n); i<h; i++){
		for (int j=w-(N-n); j<w; j++){
			sum=0;
			for (int ki=0; ki<N-(i-h+n+1); ki++ ){
				for (int kj=0; kj<N-(j-w+n+1); kj++){
					sum += K[ki*N+kj];
				}
			}
			norm=1.0/sum;
			sum=0;
			for (int u=-n; u<=h-i-1; u++){
				for (int v=-n; v<=w-j-1; v++){
					sum += im[(i+u)*h+(j+v)]*K[(u+n)*N+(v+n)]*norm;
					
				}
			}
			blur[i*h+j] = (u_int8_t)sum;
		}
	}
*/
	
	return blur;
}
