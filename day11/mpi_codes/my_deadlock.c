#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[])        
{
#define MSGLEN 2048
  int ITAG_A = 100,ITAG_B = 200; 
  int irank, i, idest, isrc, istag, iretag;
  int rmsg1[MSGLEN];
  int rmsg2[MSGLEN];
  MPI_Status recv_status;
  MPI_Request REQ;

  MPI_Init(&argc, &argv);
  //printf("argc = %d, argv[1]= %s\n", argc, argv[1]);
  MPI_Comm_rank(MPI_COMM_WORLD, &irank);  

  for (i = 0; i < MSGLEN; i++)
    {
      rmsg1[i] = 111;
      rmsg2[i] = 222;
    }
  if ( irank == 0 )
    { 
      idest  = 1;
      isrc   = 1;
      istag  = ITAG_A;
      iretag = ITAG_B;
      //rmsg1[0] = 182;
    }
  else if ( irank == 1 )
    {
      idest  = 0;
      isrc   = 0;
      istag  = ITAG_B;
      iretag = ITAG_A;
      //rmsg1[0] = -182;
    }
  printf("BEFORE: proc %d has rmsg1[0]=%d and rmsg2[0]=%d\n",irank, rmsg1[0], rmsg2[0]);   
  printf("Task %d (with tag=%d) has sent the message\n", irank, istag);
  
  MPI_Isend(&rmsg1, MSGLEN, MPI_FLOAT, idest, istag, MPI_COMM_WORLD,&REQ);
  MPI_Recv(&rmsg2, MSGLEN, MPI_FLOAT, isrc, iretag, MPI_COMM_WORLD, &recv_status);
  MPI_Wait( &REQ, &recv_status );

  /* //con Bsend non riesce a riceve e da errori
  MPI_Bsend(&rmsg1, MSGLEN, MPI_FLOAT, idest, istag, MPI_COMM_WORLD);
  MPI_Recv(&rmsg2, MSGLEN, MPI_FLOAT, isrc, iretag, MPI_COMM_WORLD, &recv_status);
  */
  
  //con Ssend si pianta dopo aver inviato e rimane li.
  
  printf("Task %d has received the message\n", irank);
  printf("AFTER: proc %d has rmsg1[0]=%d and rmsg2[0]=%d\n",irank, rmsg1[0], rmsg2[0]);   
  MPI_Finalize();
}
