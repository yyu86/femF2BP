#!/bin/bash
#
#  Compile the program with G++
#
#  Declare the number of threads and the name of PAR file
#
echo "Run with 20 threads."
export OMP_NUM_THREADS=20
./F2Bdyn run.par
#
#
echo "Completed! "
