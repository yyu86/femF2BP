#include "kinematics.h"

void QuatSet(Quaternion q, double a0, double a1, double a2, double a3)
{
    q[0] = a0;
    q[1] = a1;
    q[2] = a2;
    q[3] = a3;
}

void QuatCopy(const Quaternion q1, Quaternion q2)
{
    q2[0] = q1[0];
    q2[1] = q1[1];
    q2[2] = q1[2];
    q2[3] = q1[3];
}

void QuatNorm(Quaternion q)
{
    double mag = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    assert(mag > 0.0);
    for (int i=0; i<4; i++) q[i] = q[i] / mag;
}

/*
 Quat2DCM: transfer quaternion to direction cosine matrix
 Quat2IvDCM: transfer quaternion to the inverse of direction cosine matrix
             DCM - direction cosine matrix
             the input q must be unit quaternion
*/

void Quat2DCM(const Quaternion q, Matrix m)
{
    double qq[10];
    //
    qq[0] = q[0] * q[0];
    qq[1] = q[1] * q[1];
    qq[2] = q[2] * q[2];
    qq[3] = q[3] * q[3];
    qq[4] = 2.0 * q[0] * q[1];
    qq[5] = 2.0 * q[0] * q[2];
    qq[6] = 2.0 * q[0] * q[3];
    qq[7] = 2.0 * q[1] * q[2];
    qq[8] = 2.0 * q[1] * q[3];
    qq[9] = 2.0 * q[2] * q[3];
    //
    m[0][0] = qq[0] + qq[1] - qq[2] - qq[3];
    m[0][1] = qq[7] + qq[6];
    m[0][2] = qq[8] - qq[5];
    m[1][0] = qq[7] - qq[6];
    m[1][1] = qq[0] - qq[1] + qq[2] - qq[3];
    m[1][2] = qq[9] + qq[4];
    m[2][0] = qq[8] + qq[5];
    m[2][1] = qq[9] - qq[4];
    m[2][2] = qq[0] - qq[1] - qq[2] + qq[3];
}

void Quat2IvDCM(const Quaternion q, Matrix m)
{
    double qq[10];
    //
    qq[0] = q[0] * q[0];
    qq[1] = q[1] * q[1];
    qq[2] = q[2] * q[2];
    qq[3] = q[3] * q[3];
    qq[4] = 2.0 * q[0] * q[1];
    qq[5] = 2.0 * q[0] * q[2];
    qq[6] = 2.0 * q[0] * q[3];
    qq[7] = 2.0 * q[1] * q[2];
    qq[8] = 2.0 * q[1] * q[3];
    qq[9] = 2.0 * q[2] * q[3];
    //
    m[0][0] = qq[0] + qq[1] - qq[2] - qq[3];
    m[0][1] = qq[7] - qq[6];
    m[0][2] = qq[8] + qq[5];
    m[1][0] = qq[7] + qq[6];
    m[1][1] = qq[0] - qq[1] + qq[2] - qq[3];
    m[1][2] = qq[9] - qq[4];
    m[2][0] = qq[8] - qq[5];
    m[2][1] = qq[9] + qq[4];
    m[2][2] = qq[0] - qq[1] - qq[2] + qq[3];
}

/*
 Euler2Quat: transfer Euler angles to unit quaternion
 Euler angle uses the definition of 3-1-3 rotation:
 input a[0] = phi is the precession about z-axis
       a[1] = theta is the nutation about x'-axis
       a[2] = varphi is the spin about z"-axis
 */

void Euler2Quat(const double a[3], Quaternion q)
{
    double Cp, Sp, Ct, St, Cv, Sv;
    
    Cp = cos(a[0]/2.0);
    Sp = sin(a[0]/2.0);
    Ct = cos(a[1]/2.0);
    St = sin(a[1]/2.0);
    Cv = cos(a[2]/2.0);
    Sv = sin(a[2]/2.0);
    //
    q[0] = Cp*Ct*Cv - Sp*Ct*Sv;
    q[1] = Cp*St*Cv + Sp*St*Sv;
    q[2] = Sp*St*Cv - Cp*St*Sv;
    q[3] = Sp*Ct*Cv + Cp*Ct*Sv;
    //
    QuatNorm(q);
}

//

void getMomentumMoment(FEM &alpha, FEM &beta, PHASE ph, Vector MM)
{
    Vector MMT_A, MMR_A, MMT_B, MMR_B;
    Vector tmpV1;
    Matrix IvDCM_A, IvDCM_B;
    
    // translational moment of momentum of primary (OXYZ)
    vectorCross(ph.AlphaPos, ph.AlphaVel, tmpV1);
    vectorScale(tmpV1, alpha.TotalMass, MMT_A);
    // translational moment of momentum of secondary (OXYZ)
    vectorCross(ph.BetaPos, ph.BetaVel, tmpV1);
    vectorScale(tmpV1, beta.TotalMass, MMT_B);
    //
    // IvDCM_A: the inverse direction cosine matrix (transfer from AXaYaZa to AXYZ)
    Quat2IvDCM(ph.AlphaOrien, IvDCM_A);
    // IvDCM_B: the inverse direction cosine matrix (transfer from BXbYbZb to BXYZ)
    Quat2IvDCM(ph.BetaOrien, IvDCM_B);
    // rotational moment of momentum of primary (AXaYaZa)
    vectorMultiply(alpha.InertVec, ph.AlphaAngVel, tmpV1);
    // rotational moment of momentum of primary (AXYZ)
    vectorTransform(IvDCM_A, tmpV1, MMR_A);
    // rotational moment of momentum of secondary (BXbYbZb)
    vectorMultiply(beta.InertVec, ph.BetaAngVel, tmpV1);
    // rotational moment of momentum of secondary (BXYZ)
    vectorTransform(IvDCM_B, tmpV1, MMR_B);
    //
    vectorAdd(MMT_A, MMT_B, tmpV1);
    vectorAdd(MMR_A, MMR_B, MM);
    //
    vectorAdd(tmpV1, MM, MM); // in OXYZ
    
}

void getMomentum(FEM &alpha, FEM &beta, PHASE ph, Vector MT)
{
    Vector tmpV1;
    
    vectorScale(ph.AlphaVel, alpha.TotalMass, tmpV1);
    vectorScale(ph.BetaVel, beta.TotalMass, MT);
    //
    vectorAdd(tmpV1, MT, MT); // in OXYZ
}


void testConserv(double y[26], FEM &alpha, FEM &beta, Vector MomentumMoment, Vector Momentum)
{
    int i;
    PHASE x;
    Vector tmpV1, tmpV2;
    
    for (i=0; i<3; i++)
    {
        x.AlphaPos[i] = y[i];
        x.AlphaVel[i] = y[i+3];
        x.AlphaOrien[i] = y[i+6];
        x.AlphaAngVel[i] = y[i+10];
        x.BetaPos[i] = y[i+13];
        x.BetaVel[i] = y[i+16];
        x.BetaOrien[i] = y[i+19];
        x.BetaAngVel[i] = y[i+23];
    }
    x.AlphaOrien[3] = y[9];
    x.BetaOrien[3] = y[22];
    
    getMomentumMoment(alpha, beta, x, tmpV1);
    getMomentum(alpha, beta, x, tmpV2);
    //
    cout<<setiosflags(ios::scientific)<<setprecision(22);
//    cout<<"|                                   Total Angular Momentum                                   |                                     Total Momentum                                     |"<<endl;
    cout<<"| "<<tmpV1[0]<<" "<<tmpV1[1]<<" "<<tmpV1[2]<<" | "<<tmpV2[0]<<" "<<tmpV2[1]<<" "<<tmpV2[2]<<" | "<<endl;

    MomentumMoment[0] = tmpV1[0];
    MomentumMoment[1] = tmpV1[1];
    MomentumMoment[2] = tmpV1[2];

    Momentum[0] = tmpV2[0];
    Momentum[1] = tmpV2[1];
    Momentum[2] = tmpV2[2];

        // ofstream resfile;
    
//  Yang: output setting, the output file name is BinaryDance.bt----------consert
    // const char *outfile1 = "consert.txt";
    
        
    // resfile.open(outfile1);
    // resfile<<setiosflags(ios::scientific)<<setprecision(PrecDouble); // output precision and format
        // resfile<<setw(WidthDouble)<<t;
// resfile<<"| "<<tmpV1[0]<<" "<<tmpV1[1]<<" "<<tmpV1[2]<<" | "<<tmpV2[0]<<" "<<tmpV2[1]<<" "<<tmpV2[2]<<" | "<<endl;
    // resfile<<endl;
}










