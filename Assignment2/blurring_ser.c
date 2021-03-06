#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// gcc bluring_serial.c -lm
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

void name_gen(char* fname, int N, float f, int k_type, char* NAME);


int main(int argc ,char **argv){

	if(argc<4){
		printf("ERROR: \nYou must provide 4 arguments in executions:\n file_name.pgm,  kernel dimension, kernel case number (0 for mean, 1 for weight, 2 or gaussian), the parameter f (only if you choose the weight kernel).\n");
		exit(1);
	}


	char* filename = argv[1];
	int width=0,height=0,maxval=0;
	void* im;
	read_pgm_image( &im, &maxval, &width, &height, filename);
	swap_image( im, width, height, maxval );
	//printf("HEADER:  width=%d, height=%d, maxval=%d\n",width,height, maxval);
	//swap_image( im, width, height, maxval );
	//write_pgm_image( im, maxval, width, height, "prova.pgm");

	
	int N=strtol(argv[2], NULL, 10);
	int k_type=strtol(argv[3],NULL,10);
	float f=0;
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
		if (argc<5){
			printf("ERROR:\n You have choosen the weight kernel, you must provide f.\n");
			exit(1);
		}
		f=strtof(argv[4],NULL);
		if(f<0 || f>1){
			printf("ERROR: \nf must be in the interval [0,1]. \n");
			exit(1);
		}
	}
	float* K= kernel(k_type, f, N);

	/*
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			printf("%f ",K[i*N+j]);
		}
		printf("\n");
	}*/
	void* blur=malloc(height*width*sizeof(u_int16_t));
	bluring(K,blur,im,N,height,width);


	char final_name[42]="";
	name_gen(filename, N, f, k_type, final_name);
	
	swap_image( blur, width, height, maxval );
	write_pgm_image( blur, maxval, width, height, final_name);
	
	free(im);
	free(K);
	free(blur);
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

			
void bluring(float* K, u_int16_t* blur, u_int16_t* im, int N, int h, int w){
	int n=N/2;
	float norm, sum;
	
	int e=n,f=-n,g=n,l=-n,bool;
	for (int i=0; i<h; i++){
		for (int j=0; j<w; j++){
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




void name_gen(char* fname, int N, float f, int k_type, char* NAME){

	char* temp;
	temp = strchr(fname,'.');
	*temp = '\0';

	strcat(NAME, fname);
	strcat(NAME,".b_");

	char KN[4];
	sprintf( KN, "%d", k_type );
	strcat(NAME,KN);
	strcat(NAME,"_");

	char SN[4];
	sprintf( SN, "%d", N );
	strcat(NAME,SN);
	strcat(NAME,"x");
	strcat(NAME,SN);

	if(k_type==1){
		if(f==1 || f==0){
			strcat(NAME, "_");
			char sf[1];
			gcvt(f,1,sf);
			strcat(NAME,sf);
		}
		else{
			strcat(NAME, "_");
			char sf[2];
			int F=(f*10);
			sprintf( sf, "%d", F );
			strcat(NAME,"0");
			strcat(NAME,sf);
		}
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
