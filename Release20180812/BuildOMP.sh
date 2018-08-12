#!/bin/bash
#
#  Compile the program with G++
#
cd src/
/usr/bin/g++ -fopenmp -o test main.cpp binary.cpp body.cpp initialization.cpp kinematics.cpp mat3d.cpp -lm
#
mv test ../F2Bdyn
#
