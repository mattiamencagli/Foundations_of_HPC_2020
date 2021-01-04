#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

// mpicc bluring_mpi.c -lm
// mpirun -np 4 ./a.out check_me.pgm 9 2 0

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

void bluring(float* K,u_int16_t* blur, u_int16_t* im, int N, int h, int w, int a, int b, int c, int d);

void grid(int h, int w, int N, int myid, int numprocs, int* lw, int* lh, int* a, int* b, int* c, int* d);

void send_to_master(u_int16_t* blur, u_int16_t* lblur, int N, int h, int w, int lh, int lw,  int myid, int numprocs);

void send_to_slaves(u_int16_t* im, u_int16_t* lim, int N, int h, int w, int myid, int numprocs, int* lh, int* lw, int* a, int* b, int* c, int *d);

void name_gen(char* fname, int N, float f, int k_type, char* NAME);


int main(int argc ,char **argv){

	int myid , numprocs ;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	//clock_t start=clock();
	double start = MPI_Wtime();
	if(myid==0){
		printf("##### %d processors\n",numprocs);
	}

	if(argc<4){
		if(myid==0){
			printf("ERROR: \nYou must provide 4 arguments in executions:\n file_name.pgm,  kernel dimension, kernel case number (0 for mean, 1 for weight, 2 or gaussian), the parameter f (only if you choose the weight kernel).\n");
		}
		exit(1);
	}
	
	char* filename = argv[1];
	int N=strtol(argv[2], NULL, 10);
	int k_type=strtol(argv[3],NULL,10);
	float f=0;
	if(N<=0 || N%2==0){
		if(myid==0){
			printf("ERROR: \nThe dimension of the kernel should be a positive and odd integer.\n");
		}
		exit(1);
	}
	if( k_type!=0 && k_type!=1 && k_type!=2 ){
		if(myid==0){
			printf("ERROR: \nThe kernel case number must be: \n");
			printf("0 for mean kernel \n1 for weight kernel \n2 for gauss kernel.\n");
		}
		exit(1);
	}
	if(k_type==1){
		if (argc<5){
			if(myid==0){
				printf("ERROR:\n You have choosen the weight kernel, you must provide f.\n");
			}
			exit(1);
		}
		f=strtof(argv[4],NULL);
		if(f<0 || f>1){
			if(myid==0){
				printf("ERROR: \nf must be in the interval [0,1]. \n");
			}
			exit(1);
		}
	}
	float* K= kernel(k_type, f, N);

	void* im;
	int width=0,height=0,maxval=0;
	if(myid==0){
		read_pgm_image( &im, &maxval, &width, &height, filename);
		swap_image( im, width, height, maxval );
	}
	void* lim;
	int loc_h, loc_w, a, b, c, d;
	send_to_slaves(im, lim, N, height, width, myid, numprocs, &loc_h, &loc_w, &a, &b, &c, &d);
	free(im);
	printf("OOOOOOOOKKKKKKKKK\n\n");
	
	void* lblur=malloc((loc_h-N)*(loc_w-N)*sizeof(u_int16_t));
	bluring(K,lblur,lim,N,loc_h,loc_w,a,b,c,d);
	free(lim);
	free(K);
	
	void* blur;
	if(myid==0){
		blur=malloc(height*width*sizeof(u_int16_t));
	}
	send_to_master( blur, lblur,N, height, width, loc_h, loc_w, myid, numprocs);
	free(lblur);

	if(myid==0){
		char final_name[42]="";
		name_gen(filename, N, f, k_type, final_name);
		
		swap_image( blur, width, height, maxval );
		write_pgm_image( blur, maxval, width, height, final_name);
		free(blur);
	}
	
	double end = MPI_Wtime();
	//clock_t end=clock();
	//double tot_time= (double)(end - start)/CLOCKS_PER_SEC;
	printf("%3d: %12.8f\n",myid,end-start);
	

	MPI_Finalize();
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

			
void bluring(float* K,u_int16_t* blur, u_int16_t* im, int N, int h, int w, int a, int b, int c, int d){
	int n=N/2;
	float norm, sum;
	int N1=0, N2=0, N3=0, N4=0;
	if(a!=0){
		N1=n;
	}
	if(b!=h){
		N2=n;
	}
	if(c!=0){
		N3=n;
	}
	if(d!=w){
		N4=n;
	}
	int fw=w-N3-N4;
	
	int e=-n,f=n,g=-n,l=n, bool;
	for (int i=0+N1; i<h-N2; i++){
		for (int j=0+N3; j<w-N4; j++){
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
			blur[i*fw+j] = (u_int16_t)sum;
		}
	}
}

void grid(int h, int w, int N, int myid, int numprocs, int* lw, int* lh, int* a, int* b, int* c, int* d){

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

	int N1=0, N2=0, N3=0, N4=0;
	int n=N/2;
	*a=(myid-myid%proc_w)*(*lh)/proc_w;
	*b=(*a)+(*lh)+x_h;
	*c=(myid%proc_w)*(*lw);
	*d=(*c)+(*lw)+x_w;
	if ((*a)!=0){
		*a -= n;
		N1=n;
	}
	if (*b!=h){
		*b += n;
		N2=n;
	}
	if (*c!=0){
		*c -= n;
		N3=n;
	}
	if (*d!=w){
		*d += n;
		N4=n;
	}
	*lw+=x_w+N3+N4;
	*lh+=x_h+N1+N2;

	//printf("id=%d;\t lw=%d, lh=%d, a=%d, b=%d, c=%d, d=%d\n ",myid,*lw,*lh,*a,*b,*c,*d);
	
}



void send_to_master(u_int16_t* blur, u_int16_t* lblur,int N, int h, int w, int lh, int lw, int myid, int numprocs){
	MPI_Status status;
	int tag=42;
	int n=N/2;

	if(myid==0){
		for (int i=0; i<lh-N; i++){
			for(int j=0; j<lw-N; j++){
				blur[i*w+j]=lblur[i*lw+j];
			}
		}
		int a,b,c,d,N1,N2,N3,N4;
		for (int proc=1; proc<numprocs ; proc++) {
			N1=0;
			N2=0;
			N3=0;
			N4=0;
			grid(h,w,N,proc,numprocs,&lw,&lh,&a,&b,&c,&d);
			MPI_Recv(lblur,lw*lh,MPI_UNSIGNED_SHORT,proc,tag,MPI_COMM_WORLD,&status);
			if(a!=0){
				N1=n;
			}
			if(b!=h){
				N1=n;
			}
			if(c!=0){
				N1=n;
			}
			if(d!=w){
				N1=n;
			}
			for (int i=a+N1; i<b-N2; i++){
				for(int j=c+N3; j<d-N4; j++){
					blur[i*w+j]=lblur[(i-a)*lw+(j-c)];
				}
			}
		}
	} else {
		MPI_Send(lblur , lw*lh ,MPI_UNSIGNED_SHORT, 0 , tag ,MPI_COMM_WORLD) ;
	}
}

void send_to_slaves(u_int16_t* im, u_int16_t* lim, int N, int h, int w, int myid, int numprocs, int* lh, int* lw, int* a, int* b, int* c, int *d){
	MPI_Status status;
	int tag=42, tag1=1, tag2=2, tag3=3, tag4=4, tag5=5, tag6=6;

	if(myid==0){
		for (int proc=1; proc<numprocs ; proc++) {
			grid(h,w,N,proc,numprocs,lw,lh,a,b,c,d);
			printf("%d, %d, %d, %d,   %d, %d\n",*a,*b,*c,*d, *lh, *lw);
			lim=malloc((*lw)*(*lh)*sizeof(u_int16_t));
			for (int i=*a; i<*b; i++){
				for(int j=*c; j<*d; j++){
					lim[(i-(*a))*(*lw)+(j-(*c))]=im[i*w+j];
					//printf("lim: %d,   im: %d\n",(i-(*a))*(*lw)+(j-(*c)),i*w+j);
					printf("i=%d, j=%d\n",i,j);
				}
			}
			printf("ciao");
			MPI_Send(lw, 1, MPI_INTEGER, proc, tag1, MPI_COMM_WORLD);
			MPI_Send(lh, 1, MPI_INTEGER, proc, tag2, MPI_COMM_WORLD);
			MPI_Send(a , 1, MPI_INTEGER, proc, tag3, MPI_COMM_WORLD);
			MPI_Send(b , 1, MPI_INTEGER, proc, tag4, MPI_COMM_WORLD);
			MPI_Send(c , 1, MPI_INTEGER, proc, tag5, MPI_COMM_WORLD);
			MPI_Send(d , 1, MPI_INTEGER, proc, tag6, MPI_COMM_WORLD);
			MPI_Send(lim , (*lw)*(*lh) ,MPI_UNSIGNED_SHORT, proc, tag ,MPI_COMM_WORLD) ;
		}
		grid(h,w,N,myid,numprocs,lw,lh,a,b,c,d);
		lim=malloc((*lw)*(*lh)*sizeof(u_int16_t));
		for (int i=*a; i<*b; i++){
			for(int j=*c; j<*d; j++){
				lim[(i-(*a))*(*lw)+(j-(*c))]=im[i*w+j];
			}
		}
	} else {
		MPI_Recv(lw, 1, MPI_INTEGER, 0, tag1, MPI_COMM_WORLD, &status);
		MPI_Recv(lh, 1, MPI_INTEGER, 0, tag2, MPI_COMM_WORLD, &status);
		MPI_Recv(a , 1, MPI_INTEGER, 0, tag3, MPI_COMM_WORLD, &status);
		MPI_Recv(b , 1, MPI_INTEGER, 0, tag4, MPI_COMM_WORLD, &status);
		MPI_Recv(c , 1, MPI_INTEGER, 0, tag5, MPI_COMM_WORLD, &status);
		MPI_Recv(d , 1, MPI_INTEGER, 0, tag6, MPI_COMM_WORLD, &status);
		MPI_Recv(lim,(*lw)*(*lh),MPI_UNSIGNED_SHORT,0,tag,MPI_COMM_WORLD,&status);
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
