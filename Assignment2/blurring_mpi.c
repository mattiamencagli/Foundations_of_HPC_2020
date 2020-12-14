#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc ,char **argv){

	//###### LETTURA ######################################################
	char* filename = argv[1];
	FILE* f=fopen(filename,"rb");
    if (!f) {
        printf("Error: Unable to open file %s.\n\n", filename);
        exit(1);
    }
    printf("opening %s\n",filename);

    char type[2];
	int width=0,height=0,maxval=0;
	fscanf(f,"%2s %d %d %d",type,&width,&height,&maxval);
	printf("HEADER:  type=%s, width=%d, height=%d, maxval=%d\n",type,width,height, maxval);

	u_int8_t* im=(u_int8_t *)malloc(height*width*sizeof(u_int8_t));
	fread(im,sizeof(u_int8_t), width*height,f);     
	fclose(f);
	/*
	FILE* ff=fopen("prova.pgm","wb");
	fprintf(ff,"%2s\n%d %d\n%d\n", type, width, height, maxval);
	fwrite(im,sizeof(u_int8_t), width*height, ff);
    fclose(ff);
	*/

	//###### BLURING ######################################################
	u_int8_t* blur=(u_int8_t *)malloc(height*width*sizeof(u_int8_t));
	int N=9;
	int n=N/2;
	float a=1.0/((float)N*(float)N);
	float* K=(float *)malloc(height*width*sizeof(float));
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
			K[i*N+j]=a;
		}
	}
	
	for (int i=N-1; i<height-N+1; i++){
		for (int j=N-1; j<width-N+1; j++){
			float sum=0;
			for (int u=-n; u<=n; u++){
				for (int v=-n; v<=n; v++){
					sum += im[(i+u)*height+(j+v)]*K[(u+n)*N+(v+n)];
					
				}
			}
			blur[i*height+j] = (u_int8_t)sum;
		}
	}
/*
	for (int i=0; i<height; i++){
		for (int j=0; j<width; j++){
			printf("%d ",blur[i*height+j]);
		}
		printf("\n");
	}*/
	/*
	char fname[15]="blur_mean_X.pgm";
	fname[10]=N+'0';
	printf("Output: %s \n",fname);
	FILE* ff=fopen(fname,"wb");
    */
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

	//TODO: main program
	
	free(im);
	free(K);
	free(blur);
	return 0;
}

