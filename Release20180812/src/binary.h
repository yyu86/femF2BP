//
//  binary.h
//  
//
//  Created by Yang Yu on 10/15/17.
//
//

#ifndef _BINARY_H_
#define _BINARY_H_

#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include "constant.h"
#include "kinematics.h"
#include <omp.h>

typedef struct ode_option {

    double AbsTol;
    double RelTol;
    double TimeEnd;
    
} ODE_OPTION; // ode solver options

typedef struct mascon {
    
    double **alpha, **beta;
    
} MASCON; // ode solver options


void Simulation(FEM &alpha, FEM &beta, PHASE &ph0, ODE_OPTION &odeopt);

void MotionEquation(double t, const double* y, double* yp, FEM &alpha, FEM &beta, MASCON &varpool);

void FixNorm(double* y);

namespace rk78
{
    void RKF78(void (*Model)(double t, const double* y, double* yp, FEM &alpha, FEM &beta, MASCON &varpool), double* y, FEM &alpha, FEM &beta, MASCON &varpool, double& t, const double& tout, int dim, double RelTol, double* AbsTol, int& RKFiflag, int RKFmaxnfe);
}

using namespace std;

#endif /* defined(____binary__) */
