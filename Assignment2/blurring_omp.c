#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// gcc bluring_omp.c -lm -fopenmp
// ./a.out check_me.pgm 9 2 0

#if ((0x100 & 0xf) == 0x0)
#define I_M_LITTLE_ENDIAN 1
#define swap(mem) (( (mem) & (short int)0xff00) >> 8) +	\
  ( ((mem) & (short int)0x00ff) << 8)
#else
#define I_M_LITTLE_ENDIAN 0
#define swap(mem) (mem)
#endif

void read_pgm_image( void **image, int *maxval, int *xsize, int *ysize, const char *image_name);

void write_pgm_image( void *image, int maxval, int xsize, int ysize, const char *image_name);

void swap_image( void *image, int xsize, int ysize, int maxval );


float* kernel(int k_type, float f, int N);

void bluring(float* K,u_int16_t* blur, u_int16_t* im, int N, int h, int w);

void grid(int h, int w, int myid, int numprocs, int* lw, int* lh, int* a, int* b, int* c, int* d);

void name_gen(char* fname, int N, float f, int k_type, char* NAME);


int main(int argc ,char **argv){

	clock_t start=clock();

	if( argc<5 ){
		printf("ERROR: \nYou must provide 5 arguments in executions:\n the number of threads you want to use, file_name.pgm,  kernel dimension, kernel case number (0 for mean, 1 for weight, 2 or gaussian), the parameter f (only if you choose the weight kernel).\n");
		exit(1);
	}


	char* filename = argv[2];
	int width=0,height=0,maxval=0;
	void* im;
	read_pgm_image( &im, &maxval, &width, &height, filename);
	swap_image( im, width, height, maxval );
	//printf("HEADER:  width=%d, height=%d, maxval=%d\n",width,height, maxval);
	//swap_image( im, width, height, maxval );
	//write_pgm_image( im, maxval, width, height, "prova.pgm");

	int th_num=strtol(argv[1],NULL, 10);
	printf("##### %d threads\n",th_num);
	int N=strtol(argv[3], NULL, 10);
	int k_type=strtol(argv[4],NULL,10);
	float f=0;
	if(th_num<=0){
		printf("ERROR: \nThe number of threads must be a positive integer.\n");
		exit(1);
	}
	if(N<=0 || N%2==0){
		printf("ERROR: \nThe dimension of the kernel should be a positive and odd integer.\n");
		exit(1);
	}
	if( k_type!=0 && k_type!=1 && k_type!=2 ){
		printf("ERROR: \nThe kernel case number must be: \n");
		printf("0 for mean kernel \n1 for weight kernel \n2 for gauss kernel.\n");
		exit(1);
	}
	if(k_type==1){
		if (argc<6){
			printf("ERROR:\n You have choosen the weight kernel, you must provide f.\n");
			exit(1);
		}
		f=strtof(argv[5],NULL);
		if(f<0 || f>1){
			printf("ERROR: \nf must be in the interval [0,1]. \n");
			exit(1);
		}
	}
	float* K= kernel(k_type, f, N);
	
	void* blur=malloc(height*width*sizeof(u_int16_t));
	
	omp_set_num_threads(th_num);
	bluring(K,blur,im,N,height,width);

	
	char final_name[42]="";
	name_gen(filename, N, f, k_type, final_name);
	
	swap_image( blur, width, height, maxval );
	write_pgm_image( blur, maxval, width, height, final_name);
	
	free(blur);
	free(im);
	free(K);
	
	clock_t end=clock();
	double tot_time= (double)(end - start)/(CLOCKS_PER_SEC*th_num);
	printf("%12.8f\n",tot_time);
	return 0;
}



float * kernel(int k_type,float f, int N){
	
	float* k=(float *)malloc(N*N*sizeof(float));
	int n=N/2;
	float sum;
	
	switch(k_type){
		
		case(0): // MEAN (normalizzato)
			for (int i=0; i<N*N; i++){
				k[i]=1./((float)N*(float)N);
			}
			return k;
			
		case(1): // WEIGHT (normalizzato)
			for (int i=0; i<N*N; i++){
				k[i]=(1.-f)/((float)N*(float)N-1.);
			}
			k[(N/2)*(N+1)]=f;
			return k;
			
		case(2): // GAUSSIAN (normalizzato)
			sum=0;
			for (int i=0; i<N; i++){
				for (int j=0; j<N; j++){
					k[i*N+j]=pow(2.7182,(-(pow(i-n,2)+pow(j-n,2))/(2.*(float)n*(float)n)))/(2.*3.1415*(float)n*(float)n);
					sum +=k[i*N+j];
				}
			}
			for (int i=0; i<N*N; i++){
				k[i] /= sum;
			}
			return k;
			
	}
}

			
void bluring(float* K,u_int16_t* blur, u_int16_t* im, int N, int h, int w){
	#pragma omp parallel
	{
		int n=N/2;
		float norm, sum;
		
		int myid=omp_get_thread_num();
		int numthreads=omp_get_num_threads();

		int lw,lh,a,b,c,d;
		grid(h,w,myid,numthreads,&lw,&lh,&a,&b,&c,&d);
		
		int e=-n,f=n,g=-n,l=n,bool;
		for (int i=a; i<b; i++){
			for (int j=c; j<d; j++){
				bool=1;
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
						bool=0;
						g=-n;
						l=n;
					}	
				}
				norm=1;
				if(bool==1){ //normalization for borders
					sum=0;
					for (int u=e; u<=f; u++){
						for (int v=g; v<=l; v++){
							sum += K[(u+n)*N+(v+n)];
						}
					}
					norm=1./sum;
				}
				sum=0;
				for (int u=e; u<=f; u++){
					for (int v=g; v<=l; v++){
						sum += im[(i+u)*w+(j+v)]*K[(u+n)*N+(v+n)]*norm;
					}
				}
				blur[i*w+j] = (u_int16_t)sum;
			}
		}
	}
}

void grid(int h, int w, int myid, int numprocs, int* lw, int* lh, int* a, int* b, int* c, int* d){

	int big_dim, big_dim_loc, small_dim_loc;
	if(w>h){
		big_dim=w;
	} else{
		big_dim=h;
	}

	int tmp=((float)big_dim)/(sqrt(h*w/numprocs))+0.5;	

	while(1){
		if(numprocs%tmp){
			++tmp;
		}else{
			big_dim_loc=tmp;
			small_dim_loc=numprocs/tmp;
			break;
		}
	}
	
	int proc_w,proc_h;
	if(w>h){
		proc_w=big_dim_loc;
		proc_h=small_dim_loc;
	} else{
		proc_h=big_dim_loc;
		proc_w=small_dim_loc;
	}

	*lw=w/proc_w;
	*lh=h/proc_h;

	int x_w=0, x_h=0; 
	int avanzo_w=w-(*lw)*proc_w, avanzo_h=h-(*lh)*proc_h;
	
	if (avanzo_w!=0){
		if(myid%proc_w==proc_w-1){
			x_w=avanzo_w;
		}
	}
	if (avanzo_h!=0){
		if(myid%proc_h==proc_h-1){
			x_h=avanzo_h;
		}
	}

	*a=(myid-myid%proc_w)*(*lh)/proc_w;
	*b=(*a)+(*lh)+x_h;
	*c=(myid%proc_w)*(*lw);
	*d=(*c)+(*lw)+x_w;
	*lw+=x_w;
	*lh+=x_h;

	//printf("id=%d;\t lw=%d, lh=%d, a=%d, b=%d, c=%d, d=%d\n ",myid,*lw,*lh,*a,*b,*c,*d);
	
}


void name_gen(char* fname, int N, float f, int k_type, char* NAME){

	char* temp;
	temp = strchr(fname,'.');
	*temp = '\0';

	strcat(NAME, fname);
	strcat(NAME,"-omp_");

	char* type;
	switch(k_type){
		case(0):
			type="mean";
			break;
		case(1):
			type="weight";
			break;
		case(2):
			type="gauss";
			break;
	}
	
	strcat(NAME,type);
	strcat(NAME, "_N");

	char SN[3];
	sprintf( SN, "%d", N );
	strcat(NAME,SN);

	if(k_type==1){
		strcat(NAME, "_f");
		char sf[3];
		gcvt(f,3,sf);
		strcat(NAME,sf);
	}
	
	strcat(NAME,".pgm");
}






void write_pgm_image( void *image, int maxval, int xsize, int ysize, const char *image_name)
/*
 * image        : a pointer to the memory region that contains the image
 * maxval       : either 255 or 65536
 * xsize, ysize : x and y dimensions of the image
 * image_name   : the name of the file to be written
 *
 */
{
  FILE* image_file; 
  image_file = fopen(image_name, "w"); 
  
  // Writing header
  // The header's format is as follows, all in ASCII.
  // "whitespace" is either a blank or a TAB or a CF or a LF
  // - The Magic Number (see below the magic numbers)
  // - the image's width
  // - the height
  // - a white space
  // - the image's height
  // - a whitespace
  // - the maximum color value, which must be between 0 and 65535
  //
  // if he maximum color value is in the range [0-255], then
  // a pixel will be expressed by a single byte; if the maximum is
  // larger than 255, then 2 bytes will be needed for each pixel
  //

  int color_depth = 1 + ( maxval > 255 );

  fprintf(image_file, "P5\n# generated by\n# Mattia Mencagli\n%d %d\n%d\n", xsize, ysize, maxval);
  
  // Writing file
  fwrite( image, 1, xsize*ysize*color_depth, image_file);  

  fclose(image_file); 
  return ;

  /* ---------------------------------------------------------------

     TYPE    MAGIC NUM     EXTENSION   COLOR RANGE
           ASCII  BINARY

     PBM   P1     P4       .pbm        [0-1]
     PGM   P2     P5       .pgm        [0-255]
     PPM   P3     P6       .ppm        [0-2^16[
  
  ------------------------------------------------------------------ */
}


void read_pgm_image( void **image, int *maxval, int *xsize, int *ysize, const char *image_name)
/*
 * image        : a pointer to the pointer that will contain the image
 * maxval       : a pointer to the int that will store the maximum intensity in the image
 * xsize, ysize : pointers to the x and y sizes
 * image_name   : the name of the file to be read
 *
 */
{
  FILE* image_file; 
  image_file = fopen(image_name, "r"); 

  *image = NULL;
  *xsize = *ysize = *maxval = 0;
  
  char    MagicN[2];
  char   *line = NULL;
  size_t  k, n = 0;
  
  // get the Magic Number
  k = fscanf(image_file, "%2s%*c", MagicN );

  // skip all the comments
  k = getline( &line, &n, image_file);
  while ( (k > 0) && (line[0]=='#') )
    k = getline( &line, &n, image_file);

  if (k > 0)
    {
      k = sscanf(line, "%d%*c%d%*c%d%*c", xsize, ysize, maxval);
      if ( k < 3 )
	fscanf(image_file, "%d%*c", maxval);
    }
  else
    {
      *maxval = -1;         // this is the signal that there was an I/O error
			    // while reading the image header
      free( line );
      return;
    }
  free( line );
  
  int color_depth = 1 + ( *maxval > 255 );
  unsigned int size = *xsize * *ysize * color_depth;
  
  if ( (*image = (char*)malloc( size )) == NULL )
    {
      fclose(image_file);
      *maxval = -2;         // this is the signal that memory was insufficient
      *xsize  = 0;
      *ysize  = 0;
      return;
    }
  
  if ( fread( *image, 1, size, image_file) != size )
    {
      free( image );
      image   = NULL;
      *maxval = -3;         // this is the signal that there was an i/o error
      *xsize  = 0;
      *ysize  = 0;
    }  

  fclose(image_file);
  return;
}


void swap_image( void *image, int xsize, int ysize, int maxval )
/*
 * This routine swaps the endianism of the memory area pointed
 * to by ptr, by blocks of 2 bytes
 *
 */
{
  if ( maxval > 255 )
    {
      // pgm files has the short int written in
      // big endian;
      // here we swap the content of the image from
      // one to another
      //
      unsigned int size = xsize * ysize;
      for ( int i = 0; i < size; i++ )
  	((unsigned short int*)image)[i] = swap(((unsigned short int*)image)[i]);
    }
  return;
}
