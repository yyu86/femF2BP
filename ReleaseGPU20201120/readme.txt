This code was developed to tack the orbital and librational motion of a gravitational full two-body system, using the finite element method developed for such purpose.

Operating environment: Ubuntu 18.04, CUDA 10.0

HOW TO RUN:
	$ nvcc -o binarymotion binary.cu *.cpp
	$ ./binarymotion run_didymos.par


