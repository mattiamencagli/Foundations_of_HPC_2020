# Assignment 1

## Section 1

I use the Python script `perf_mod.ipynb` to compute the algorithm and to generate the csv file and the plots presented in the report.

## Section 2

I run the serial tests using the script `ser.script`, that has the command: 
`/usr/bin/time ./pi.x ${MOVES} >out.ser 2>out.time.ser` 

I modify `mpi_pi.c` in `my_mpi.c`, here I only change the way the program print out the output: I pass all the final times of each processor to the master that have to print all of it,so the final output is perfectly ordered. In this way the output is much more simple to be read by the Python script `analisi_scaling.ipynb` that I write to generate the csv files and the plots presented in the report.

I use the module `openmpi/4.0.3/gnu/9.3.0`, and I run the mpi tests using a script ( `mpi.scrpit.*` ) with a command inside as:
`/usr/bin/time mpirun  --mca btl '^openib' -np ${procs} my_mpi_pi.x  ${MOVES} >>out.mpi.s8.1 2>>out.time.mpi.s8.1`

