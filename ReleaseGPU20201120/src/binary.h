//
//  binary.h
//  
//
//  Created by Yang Yu & Yunfeng Gao on 10/15/20.
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

void MotionEquation(double t, const double* y, double* yp, FEM &alpha, FEM &beta, MASCON &varpool, double* dx1,double* dx2,double* dx3,double* dx4,double* dx5,double* dx6,double* dwx,double* dy1,double* dy2,double* dy3,double* dy4,double* dy5,double* dy6,double* dwy,
    double* d_GFAx,double* d_GFAy,double* d_GFAz,double* d_GTAx,double* d_GTAy,double* d_GTAz,double* d_GTBx,double* d_GTBy,double* d_GTBz,
    double* d1_GFAx,double* d2_GFAx,double* d3_GFAx,double* de_GFAx,double* d1_GFAy,double* d2_GFAy,double* d3_GFAy,double* de_GFAy,double* d1_GFAz,double* d2_GFAz,double* d3_GFAz,double* de_GFAz,
    double* d1_GTAx,double* d2_GTAx,double* d3_GTAx,double* de_GTAx,double* d1_GTAy,double* d2_GTAy,double* d3_GTAy,double* de_GTAy,double* d1_GTAz,double* d2_GTAz,double* d3_GTAz,double* de_GTAz,
    double* d1_GTBx,double* d2_GTBx,double* d3_GTBx,double* de_GTBx,double* d1_GTBy,double* d2_GTBy,double* d3_GTBy,double* de_GTBy,double* d1_GTBz,double* d2_GTBz,double* d3_GTBz,double* de_GTBz);

void FixNorm(double* y);

namespace rk78
{
    void RKF78(void (*Model)(double t, const double* y, double* yp, FEM &alpha, FEM &beta, MASCON &varpool,double* dx1,double* dx2,double* dx3,double* dx4,double* dx5,double* dx6,double* dwx,double* dy1,double* dy2,double* dy3,double* dy4,double* dy5,double* dy6,double* dwy,
    double* d_GFAx,double* d_GFAy,double* d_GFAz,double* d_GTAx,double* d_GTAy,double* d_GTAz,double* d_GTBx,double* d_GTBy,double* d_GTBz,
    double* d1_GFAx,double* d2_GFAx,double* d3_GFAx,double* de_GFAx,double* d1_GFAy,double* d2_GFAy,double* d3_GFAy,double* de_GFAy,double* d1_GFAz,double* d2_GFAz,double* d3_GFAz,double* de_GFAz,
    double* d1_GTAx,double* d2_GTAx,double* d3_GTAx,double* de_GTAx,double* d1_GTAy,double* d2_GTAy,double* d3_GTAy,double* de_GTAy,double* d1_GTAz,double* d2_GTAz,double* d3_GTAz,double* de_GTAz,
    double* d1_GTBx,double* d2_GTBx,double* d3_GTBx,double* de_GTBx,double* d1_GTBy,double* d2_GTBy,double* d3_GTBy,double* de_GTBy,double* d1_GTBz,double* d2_GTBz,double* d3_GTBz,double* de_GTBz), double* y, FEM &alpha, FEM &beta, MASCON &varpool, double& t, const double& tout, int dim, double RelTol, double* AbsTol, int& RKFiflag, int RKFmaxnfe,
    double* dx1,double* dx2,double* dx3,double* dx4,double* dx5,double* dx6,double* dwx,double* dy1,double* dy2,double* dy3,double* dy4,double* dy5,double* dy6,double* dwy,
    double* d_GFAx,double* d_GFAy,double* d_GFAz,double* d_GTAx,double* d_GTAy,double* d_GTAz,double* d_GTBx,double* d_GTBy,double* d_GTBz,
    double* d1_GFAx,double* d2_GFAx,double* d3_GFAx,double* de_GFAx,double* d1_GFAy,double* d2_GFAy,double* d3_GFAy,double* de_GFAy,double* d1_GFAz,double* d2_GFAz,double* d3_GFAz,double* de_GFAz,
    double* d1_GTAx,double* d2_GTAx,double* d3_GTAx,double* de_GTAx,double* d1_GTAy,double* d2_GTAy,double* d3_GTAy,double* de_GTAy,double* d1_GTAz,double* d2_GTAz,double* d3_GTAz,double* de_GTAz,
    double* d1_GTBx,double* d2_GTBx,double* d3_GTBx,double* de_GTBx,double* d1_GTBy,double* d2_GTBy,double* d3_GTBy,double* de_GTBy,double* d1_GTBz,double* d2_GTBz,double* d3_GTBz,double* de_GTBz);
}

using namespace std;

#endif /* defined(____binary__) */
