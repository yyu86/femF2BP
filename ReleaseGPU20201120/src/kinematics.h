//
//  Created by Yang Yu & Yunfeng Gao on 10/15/20.
//
//

#ifndef _KINEMATICS_H_
#define _KINEMATICS_H_

#include "body.h"
#include "constant.h"
#include <iomanip> //just for testConserv

typedef double Quaternion[4];

void QuatSet(Quaternion q, double a0, double a1, double a2, double a3);
void QuatCopy(Quaternion q1, Quaternion q2);
void QuatNorm(Quaternion q);
void Quat2DCM(const Quaternion q, Matrix m);
void Quat2IvDCM(const Quaternion q, Matrix m);
void DCM2Quat(const Matrix m, Quaternion q);
void Euler2Quat(const double a[3], Quaternion q);

typedef struct phase {
    
    Vector AlphaPos; // OXYZ
    Vector AlphaVel; // OXYZ
    Quaternion AlphaOrien; // AXYZ->AXaYaZa
    Vector AlphaAngVel; //AXaYaZa
    //
    Vector BetaPos; // OXYZ
    Vector BetaVel; // OXYZ
    Quaternion BetaOrien; // BXYZ->BXbYbZb
    Vector BetaAngVel; // BXbYbZb
    
} PHASE;

void getMomentumMoment(FEM &alpha, FEM &beta, PHASE ph, Vector MM);
void getMomentum(FEM &alpha, FEM &beta, PHASE ph, Vector MT);
void testConserv(double y[26], FEM &alpha, FEM &beta, Vector MomentumMoment, Vector Momentum);

#endif
