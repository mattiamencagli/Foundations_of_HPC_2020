# Assignment 1

## Section 1

I use a Python script (perf_mod.ipynb) to compute the algorithm and to generate the csv file and the plots presented in the report.

## Section 2

I modify "mpi_pi.c" in "my_mpi.c", here I only change the way the program print out the output: I pass all the final times of each processor to the master that have to print all of it,so the final output is perfectly ordered.
In this way the output is much more simple to be read by the Python (analisi_scaling.ipynb) script that I write to generate the csv files and the plots presented in the report.

