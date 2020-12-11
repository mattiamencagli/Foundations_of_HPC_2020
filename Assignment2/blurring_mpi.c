#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>


int main(int argc ,char **argv){

	//TODO: load the image

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

	//TODO: main program
	
	MPI_Finalize() ;
	
	return 0;
}

int 
