//
//  binary.cpp
//  
//
//  Created by Yang Yu on 10/15/17.
//
//

#include "binary.h"

void Simulation(FEM &alpha, FEM &beta, PHASE &ph0, ODE_OPTION &odeopt)
{
    int i, RKFiflag = 1;
    double tin, tout, reltol, abstol[EqnDim], y[EqnDim];
    //
    MASCON varpool;
    
    varpool.alpha = new double*[alpha.NodeNum];
    for (i=0; i<alpha.NodeNum; i++) varpool.alpha[i] = new double[6];
    varpool.beta = new double*[beta.NodeNum];
    for (i=0; i<beta.NodeNum; i++) varpool.beta[i] = new double[6];
    //
    reltol = odeopt.RelTol;
    
    for (i=0; i<EqnDim; i++) abstol[i] = odeopt.AbsTol;
    
    tin = 0.0;
    
    tout = odeopt.TimeEnd;
    
    for (i=0; i<3; i++)
    {
        y[i] = ph0.AlphaPos[i];
        y[i+3] = ph0.AlphaVel[i];
        y[i+6] = ph0.AlphaOrien[i];
        y[i+10] = ph0.AlphaAngVel[i];
        y[i+13] = ph0.BetaPos[i];
        y[i+16] = ph0.BetaVel[i];
        y[i+19] = ph0.BetaOrien[i];
        y[i+23] = ph0.BetaAngVel[i];
    }
    y[9] = ph0.AlphaOrien[3];
    y[22] = ph0.BetaOrien[3];
    
    rk78::RKF78(MotionEquation, y, alpha, beta, varpool, tin, tout, EqnDim, reltol, abstol, RKFiflag, 100000);

}

void MotionEquation(double t, const double* y, double* yp, FEM &alpha, FEM &beta, MASCON &varpool)
{
    int i,j;
    double tmpd;
    Vector tmpV1, tmpV2;
    Vector GravForceA, GravTorqueA, GravForceB, GravTorqueB;
    Matrix DCM_A, IvDCM_A, DCM_B, IvDCM_B;
    PHASE x, dx;
    
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
    
    // Normalize the quaternions of orienation
    QuatNorm(x.AlphaOrien);
    QuatNorm(x.BetaOrien);
    //
    // IvDCM_A: the inverse direction cosine matrix (transfer from AXaYaZa to AXYZ)
    Quat2IvDCM(x.AlphaOrien, IvDCM_A);
    // IvDCM_B: the inverse direction cosine matrix (transfer from BXbYbZb to BXYZ)
    Quat2IvDCM(x.BetaOrien, IvDCM_B);
    // DCM_A: the direction cosine matrix (transfer from AXYZ to AXaYaZa)
    Quat2DCM(x.AlphaOrien, DCM_A);
    // DCM_B: the direction cosine matrix (transfer from BXYZ to BXbYbZb)
    Quat2DCM(x.BetaOrien, DCM_B);
    // zero the gravity forces and torques in OXYZ
    vectorZero(GravForceA); // attraction on Alpha
    vectorZero(GravTorqueA); // torque on Alpha
    vectorZero(GravTorqueB); // torque on Beta
    //
    // OMP~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ start ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    for (i=0; i<alpha.NodeNum; i++)
    {
        vectorSet(tmpV1,alpha.Nodes[i][0],alpha.Nodes[i][1],alpha.Nodes[i][2]);
        vectorTransform(IvDCM_A, tmpV1, tmpV2);
        vectorAdd(x.AlphaPos,tmpV2,tmpV1);
        for (j=0; j<3; j++)
        {
            varpool.alpha[i][j] = tmpV1[j];
            varpool.alpha[i][j+3] = tmpV2[j];
        }
    }
    //
    for (i=0; i<beta.NodeNum; i++)
    {
        vectorSet(tmpV1,beta.Nodes[i][0],beta.Nodes[i][1],beta.Nodes[i][2]);
        vectorTransform(IvDCM_B, tmpV1, tmpV2);
        vectorAdd(x.BetaPos,tmpV2,tmpV1);
        for (j=0; j<3; j++)
        {
            varpool.beta[i][j] = tmpV1[j];
            varpool.beta[i][j+3] = tmpV2[j];
        }
    }
    //
    int coreNum = omp_get_num_procs();
    double **GFAlist, **GTAlist, **GTBlist;
    
    GFAlist = new double*[coreNum];
    GTAlist = new double*[coreNum];
    GTBlist = new double*[coreNum];
    for (i=0; i<coreNum; i++)
    {
        GFAlist[i] = new double[3];
        GTAlist[i] = new double[3];
        GTBlist[i] = new double[3];
    }
    //
    for (i=0; i<coreNum; i++)
    {
        for (j=0; j<3; j++)
        {
            GFAlist[i][j] = 0.0;
            GTAlist[i][j] = 0.0;
            GTBlist[i][j] = 0.0;
        }
    }
    //
    # pragma omp parallel shared(varpool,alpha,beta,GFAlist,GTAlist,GTBlist)
    {
    # pragma omp for schedule (static)
    for (int ii=0; ii<alpha.NodeNum; ii++)
    {
        double dd;
        Vector VV1,VV2,VV3,VV4;
        Vector GFA,GTA,GTB;
        
        vectorZero(GFA); //
        vectorZero(GTA); //
        vectorZero(GTB); //
        vectorSet(VV2,varpool.alpha[ii][3],varpool.alpha[ii][4],varpool.alpha[ii][5]);
        for (int jj=0; jj<beta.NodeNum; jj++)
        {
            vectorSet(VV3,varpool.beta[jj][0]-varpool.alpha[ii][0],varpool.beta[jj][1]-varpool.alpha[ii][1], \
                      varpool.beta[jj][2]-varpool.alpha[ii][2]);
            dd = vectorMag(VV3);
            dd = G*alpha.NodeWeights[ii]*beta.NodeWeights[jj]/(dd*dd*dd);
            vectorScale(VV3,dd,VV1);
            vectorAdd(GFA,VV1,GFA);
            vectorCross(VV2,VV1,VV4);
            vectorAdd(GTA,VV4,GTA);
            vectorSet(VV3,varpool.beta[jj][3],varpool.beta[jj][4],varpool.beta[jj][5]);
            vectorCross(VV1,VV3,VV4);
            vectorAdd(GTB,VV4,GTB);
        }
        
        int ID = omp_get_thread_num();
        for (int kk=0; kk<3; kk++)
        {
            GFAlist[ID][kk] = GFAlist[ID][kk] + GFA[kk];
            GTAlist[ID][kk] = GTAlist[ID][kk] + GTA[kk];
            GTBlist[ID][kk] = GTBlist[ID][kk] + GTB[kk];
        }
    }
    }
    // OMP~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // OMP barrier?
    //
    for (j=0; j<3; j++)
    {
        for (i=0; i<coreNum; i++)
        {
            GravForceA[j] = GravForceA[j] + GFAlist[i][j];
            GravTorqueA[j] = GravTorqueA[j] + GTAlist[i][j];
            GravTorqueB[j] = GravTorqueB[j] + GTBlist[i][j];
        }
    }
    //????
    //cout<<GravForceA[0]<<"  "<<GravForceA[1]<<"  "<<GravForceA[2]<<endl;
    //
    vectorScale(GravForceA, -1.0, GravForceB); // attraction on Beta in OXYZ
    // torque on Beta transfered to AXaYaZa
    vectorTransform(DCM_A, GravTorqueA, GravTorqueA);
    // torque on Beta transfered to BXbYbZb
    vectorTransform(DCM_B, GravTorqueB, GravTorqueB);
    // calculate dx.AlphaPos (OXYZ)
    vectorCopy(x.AlphaVel, dx.AlphaPos);
    // calculate dx.AlphaVel (OXYZ)
    tmpd = 1.0 / alpha.TotalMass;
    vectorScale(GravForceA, tmpd, dx.AlphaVel);
    // calculate dx.AlphaOrien (AXYZ->AXaYaZa)
    dx.AlphaOrien[0] = 0.5 * (-x.AlphaOrien[1]*x.AlphaAngVel[0] - x.AlphaOrien[2]*x.AlphaAngVel[1] - x.AlphaOrien[3]*x.AlphaAngVel[2]);
    dx.AlphaOrien[1] = 0.5 * ( x.AlphaOrien[0]*x.AlphaAngVel[0] + x.AlphaOrien[2]*x.AlphaAngVel[2] - x.AlphaOrien[3]*x.AlphaAngVel[1]);
    dx.AlphaOrien[2] = 0.5 * ( x.AlphaOrien[0]*x.AlphaAngVel[1] - x.AlphaOrien[1]*x.AlphaAngVel[2] + x.AlphaOrien[3]*x.AlphaAngVel[0]);
    dx.AlphaOrien[3] = 0.5 * ( x.AlphaOrien[0]*x.AlphaAngVel[2] + x.AlphaOrien[1]*x.AlphaAngVel[1] - x.AlphaOrien[2]*x.AlphaAngVel[0]);
    // calculate dx.AlphaAngVel (AXaYaZa)
    dx.AlphaAngVel[0] = ((alpha.InertVec[1]-alpha.InertVec[2])*x.AlphaAngVel[1]*x.AlphaAngVel[2] + GravTorqueA[0]) / alpha.InertVec[0];
    dx.AlphaAngVel[1] = ((alpha.InertVec[2]-alpha.InertVec[0])*x.AlphaAngVel[0]*x.AlphaAngVel[2] + GravTorqueA[1]) / alpha.InertVec[1];
    dx.AlphaAngVel[2] = ((alpha.InertVec[0]-alpha.InertVec[1])*x.AlphaAngVel[0]*x.AlphaAngVel[1] + GravTorqueA[2]) / alpha.InertVec[2];
    // calculate dx.BetaPos (OXYZ)
    vectorCopy(x.BetaVel, dx.BetaPos);
    // calculate dx.BetaVel (OXYZ)
    tmpd = 1.0 / beta.TotalMass;
    vectorScale(GravForceB, tmpd, dx.BetaVel);
    // calculate dx.BetaOrien (BXYZ->BXbYbZb)
    dx.BetaOrien[0] = 0.5 * (-x.BetaOrien[1]*x.BetaAngVel[0] - x.BetaOrien[2]*x.BetaAngVel[1] - x.BetaOrien[3]*x.BetaAngVel[2]);
    dx.BetaOrien[1] = 0.5 * ( x.BetaOrien[0]*x.BetaAngVel[0] + x.BetaOrien[2]*x.BetaAngVel[2] - x.BetaOrien[3]*x.BetaAngVel[1]);
    dx.BetaOrien[2] = 0.5 * ( x.BetaOrien[0]*x.BetaAngVel[1] - x.BetaOrien[1]*x.BetaAngVel[2] + x.BetaOrien[3]*x.BetaAngVel[0]);
    dx.BetaOrien[3] = 0.5 * ( x.BetaOrien[0]*x.BetaAngVel[2] + x.BetaOrien[1]*x.BetaAngVel[1] - x.BetaOrien[2]*x.BetaAngVel[0]);
    // calculate dx.BetaAngVel (BXbYbZb)
    dx.BetaAngVel[0] = ((beta.InertVec[1]-beta.InertVec[2])*x.BetaAngVel[1]*x.BetaAngVel[2] + GravTorqueB[0]) / beta.InertVec[0];
    dx.BetaAngVel[1] = ((beta.InertVec[2]-beta.InertVec[0])*x.BetaAngVel[0]*x.BetaAngVel[2] + GravTorqueB[1]) / beta.InertVec[1];
    dx.BetaAngVel[2] = ((beta.InertVec[0]-beta.InertVec[1])*x.BetaAngVel[0]*x.BetaAngVel[1] + GravTorqueB[2]) / beta.InertVec[2];
    //
    for (i=0; i<3; i++)
    {
        yp[i] = dx.AlphaPos[i];
        yp[i+3] = dx.AlphaVel[i];
        yp[i+6] = dx.AlphaOrien[i];
        yp[i+10] = dx.AlphaAngVel[i];
        yp[i+13] = dx.BetaPos[i];
        yp[i+16] = dx.BetaVel[i];
        yp[i+19] = dx.BetaOrien[i];
        yp[i+23] = dx.BetaAngVel[i];
    }
    yp[9] = dx.AlphaOrien[3];
    yp[22] = dx.BetaOrien[3];
    //
}

namespace rk78{
double ArrayMax(const double* array, int dim)
{
	double temp=array[0];
	for(int i=0;i<dim;i++)
		temp=(temp<array[i])?array[i]:temp;
	return temp;
}

double ArrayMin(const double* array, int dim)
{
	double temp=array[0];
	for(int i=0;i<dim;i++)
		temp=(temp>array[i])?array[i]:temp;
	return temp;
}

double Max (double x, double y) 
{ return (x>y)?x:y; }

double Min (double x, double y) 
{ return (x<y)?x:y; }

double CopySign (double x, double y)
{ return (y < 0.0) ? ((x < 0.0) ? x : -x) : ((x > 0.0) ? x : -x); }

int CopySign (int x, int y)
{ return (y < 0.0) ? ((x < 0.0) ? x : -x) : ((x > 0.0) ? x : -x); }

//RKF78
double RKFErrorTerm(int k, double** RKFWork)
{
	//RKF78
	static const double RKFe[13]=
		{ 41.0/840.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 41.0/840.0, -41.0/840.0, -41.0/840.0 };
    double sum = 0.0;
    for (int i = 0; i <= 12; i++)
        sum += RKFe[i] * RKFWork[i][k];
    return sum;
}

//RKF78
void RKFStep(void (*Model)(double t, const double* y, double* yp, FEM &alpha, FEM &beta, MASCON &varpool), const double* y, FEM &alpha, FEM &beta, MASCON &varpool, double t, double h, double* ss, int dim, double** RKFWork)
{
	//RKF78
	static const double RKFa[13]=
		{ 0.0, 2.0/27.0,  1.0/9.0,  1.0/6.0,  5.0/12.0, 1.0/2.0, 5.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/3.0, 1.0,  0.0, 1.0};
	static const double RKFb[13][12]=
		{
			{  0.0,            0.0,       0.0,         0.0,           0.0,           0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, },
			{  2.0/27.0,       0.0,       0.0,         0.0,           0.0,           0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, },
			{  1.0/36.0,       1.0/12.0,  0.0,         0.0,           0.0,           0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, },
			{  1.0/24.0,       0.0,       1.0/8.0,     0.0,           0.0,           0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, },
			{  5.0/12.0,       0.0,       -25.0/16.0,  25.0/16.0,     0.0,           0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, },
			{  1.0/20.0,       0.0,       0.0,         1.0/4.0,       1.0/5.0,       0.0,         0.0,           0.0,       0.0,        0.0,       0.0, 0.0, },
			{  -25.0/108.0,    0.0,       0.0,         125.0/108.0,   -65.0/27.0,    125.0/54.0,  0.0,           0.0,       0.0,        0.0,       0.0, 0.0, },
			{  31.0/300.0,     0.0,       0.0,         0.0,           61.0/225.0,    -2.0/9.0,    13.0/900.0,    0.0,       0.0,        0.0,       0.0, 0.0, },
			{  2.0,            0.0,       0.0,         -53.0/6.0,     704.0/45.0,    -107.0/9.0,  67.0/90.0,     3.0,       0.0,        0.0,       0.0, 0.0, },
			{  -91.0/108.0,    0.0,       0.0,         23.0/108.0,    -976.0/135.0,  311.0/54.0,  -19.0/60.0,    17.0/6.0,  -1.0/12.0,  0.0,       0.0, 0.0, },
			{  2383.0/4100.0,  0.0,       0.0,         -341.0/164.0,  4496.0/1025.0, -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0, 0.0, 0.0, },
			{  3.0/205.0,      0.0,       0.0,         0.0,           0.0,           -6.0/41.0,   -3.0/205.0,    -3.0/41.0, 3.0/41.0,   6.0/41.0,  0.0, 0.0, },
			{  -1777.0/4100.0, 0.0,       0.0,         -341.0/164.0,  4496.0/1025.0, -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 12.0/41.0, 0.0, 1.0, }
	   };
	static const double RKFc[13]=
		{ 0.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 0.0, 41.0/840.0, 41.0/840.0};
	
    int hi=dim-1;	
	for(int j=1;j<=12;j++)
	{
        for(int i =0;i<=hi;i++)
		{
            double x = 0.0;
            for (int m = 0; m < j; m++) 
                x += RKFb[j][m] * RKFWork[m][i];
            ss[i] = x * h + y[i];
        }
        Model(t + RKFa[j] * h, ss, RKFWork[j], alpha, beta, varpool);
    }

    for (int i = 0; i <= hi; i++)
	{
        double x = 0.0;
        for (int j = 0; j <= 12; j++) 
            x += RKFc[j] * RKFWork[j][i];
        ss[i] = h * x + y[i];
    }	
}

//RKF78
void RKF78(void (*Model)(double t, const double* y, double* yp, FEM &alpha, FEM &beta, MASCON &varpool), double* y, FEM &alpha, FEM &beta, MASCON &varpool, double& t, const double& tout, int dim, double RelTol, double* AbsTol, int& RKFiflag, int RKFmaxnfe)
{
    // Get machine epsilon
 
    const double eps = 2.2204460492503131e-016,
                 u26 = 26*eps;               
    
    const double remin = 1e-15;	

    int mflag = abs(RKFiflag);
	int i, RKFnfe, RKFkop, RKFinit=0, RKFkflag=0, RKFjflag=0;
	double RKFh, RKFsavre=0.0, RKFsavae=0.0;

    ofstream resfile;
    
//  Yang: output setting, the output file name is BinaryDance.bt----------
    const char *outfile = "BinaryOutput.bt";
        
    resfile.open(outfile);
    resfile<<setiosflags(ios::scientific)<<setprecision(PrecDouble); // output precision and format

    resfile<<setw(WidthDouble)<<t;
    for(i=0; i<dim; i++) resfile<<setw(WidthDouble)<<y[i];
    resfile<<endl;
//  ----------------------------------------------------------------------
    
    if (dim < 1 || RelTol < 0.0 || ArrayMin(AbsTol, dim) < 0.0 ||  mflag == 0  
        || mflag > 8 || ((fabs(t - tout)<1.0E-14) && (RKFkflag != 3))) 
	{
        RKFiflag = 8;
        return;
    }
    
    double dt,rer,scale,hmin,eeoet,ee,et,esttol,ss,ae,tol = 0;
    int output, hfaild, k;
	double** RKFWork=new double*[14];
	for(i=0;i<14;i++) RKFWork[i]=new double[dim];
    int lo = 0, hi = dim-1;    
    int gflag = 0;
         
    if (RKFiflag == 3||(mflag == 2 && (RKFinit == 0 || RKFkflag == 2))) 
	{
        gflag = 1;
        goto next;
    }

    if (RKFiflag == 4 || (RKFkflag == 4 && mflag == 2))
	{
        RKFnfe = 0;
        if (mflag != 2) gflag = 1;
        goto next;
    }
        
    if ((RKFkflag == 5 && ArrayMin(AbsTol,dim) == 0.0)
        || (RKFkflag == 6 && RelTol < RKFsavre && ArrayMin(AbsTol,dim) < RKFsavae))
	{
        RKFiflag = 9;
        goto final;
    }
   
  next:

    if (gflag) 
	{
        RKFiflag = RKFjflag;
        if (RKFkflag == 3) mflag = abs(RKFiflag);
    }

    RKFjflag = RKFiflag;
    RKFkflag = 0;
    RKFsavre = RelTol;
    RKFsavae = ArrayMin(AbsTol,dim);   
    
    rer = 2 * eps + remin;    
    if (RelTol < rer) 
	{
    //    RelTol = rer;
        RKFiflag = RKFkflag = 3;
        cout<<"RelTol setting too small!"<<endl;
        goto final;
    }
    
    gflag = 0;
    dt = tout - t;

    if (mflag == 1)
	{
        RKFinit = 0;
        RKFkop = 0;
        gflag = 1;
        Model(t,y,RKFWork[0], alpha, beta, varpool);  // call function
        RKFnfe = 1;
        if (fabs(t - tout)<1.0E-14) 
		{
            RKFiflag = 2;
            goto final;
        }
    }

    if (RKFinit == 0 || gflag)
	{
        RKFinit = 1;
        RKFh = fabs(dt);
        double ypk;
        for (int k = lo; k <= hi; k++)
		{
            tol = Max(RelTol*fabs(y[k]), AbsTol[k]);//RelTol * fabs(y(k)) + AbsTol;//
            if (tol > 0.0) 
			{
                ypk = fabs(RKFWork[0][k]);
				double RKFh8pow=RKFh*RKFh;
                
                RKFh8pow *= RKFh8pow;
                RKFh8pow *= RKFh8pow;
				
                if (ypk * RKFh8pow > tol)
				{
					double temp=tol/ypk;					
                    RKFh = sqrt(sqrt(sqrt(temp)));
				}
            }
        }

        if (tol <= 0.0) RKFh = 0.0;
        ypk = Max(fabs(dt),fabs(t));
        RKFh = Max(RKFh, u26 * ypk);
        RKFjflag = CopySign(2,RKFiflag);
    }

    // Set stepsize for integration in the direction from t to tout

    RKFh = CopySign(RKFh,dt);

    // Test to see if this routine is being severely impacted by too many
    // output points

    if (fabs(RKFh) >= 2*fabs(dt)) RKFkop++;

    if (RKFkop == 100) {
        RKFkop = 0;
        RKFiflag = 7;
        goto final;
    }

    if (fabs(dt) <= u26 * fabs(t)) 
	{
        // If too close to output point,extrapolate and return
        for (int k = lo; k <= hi; k++)
            y[k] += dt * RKFWork[0][k];

        Model(tout,y,RKFWork[0], alpha, beta, varpool);
        RKFnfe++;
        t = tout;
        RKFiflag = 2;
        goto final;
    }

    // Initialize output point indicator

    output = false;

    // To avoid premature underflow in the error tolerance function,
    // scale the error tolerances

    scale = 2.0 / RelTol;
    ae = scale * ArrayMin(AbsTol,dim); 

    // Step by step integration - as an endless loop over steps

    for (;;) 
	{ 

        hfaild = 0;

        // Set smallest allowable stepsize

        hmin = u26 * fabs(t);
        dt = tout - t;
        if (fabs(dt) < 2.0 * fabs(RKFh))
		{
            if (fabs(dt) <= fabs(RKFh)) 
			{

                // The next successful step will complete the 
                // integration to the output point

                output = true;
                RKFh = dt;
            }
			else
                RKFh = 0.5 * dt;
        }
        
        if (RKFnfe > RKFmaxnfe)
		{
            RKFiflag = RKFkflag = 4;
            goto final;
        }
        
    step:
        
        RKFStep(Model, y, alpha, beta, varpool, t, RKFh, RKFWork[13], dim, RKFWork);
        for (i = lo; i <= hi; i++) RKFWork[1][i] = RKFWork[13][i];
        RKFnfe += 8;
        eeoet = 0.0;
        for (k = lo; k <= hi; k++)
		{

            et = fabs(y[k]) + fabs(RKFWork[1][k]) + ae;
            // Inappropriate error tolerance

            if (et <= 0.0) 
			{
                RKFiflag = 5;
                goto final;
            }
            
            ee = fabs( RKFErrorTerm(k,RKFWork));
            eeoet = Max(eeoet,ee/et);
        }
        
        esttol = fabs(RKFh) * eeoet * scale;
        
        if (esttol > 1.0)
		{
            hfaild = true;
            output = false;
            ss = 0.1;
            if (esttol < 43046721.0) ss = 0.9 / sqrt(sqrt(sqrt(esttol)));//pow(esttol, 0.125);
            RKFh *= ss;
            if (fabs(RKFh) > hmin) goto step; // loop

            // Requested error unattainable at smallest allowable stepsize

            RKFiflag = RKFkflag = 6;
            goto final;
        }

        // Successful step
        // Store solution at t+h and evaluate derivatives there

        t += RKFh;
//		cout<<t<<","<<RKFh<<endl;
        for (k = lo; k <= hi; k++) y[k] = RKFWork[1][k];

        Model(t,y,RKFWork[0], alpha, beta, varpool);
        RKFnfe++;     
        ss = 5.0;
        if (esttol > 2.565784513950347900390625E-8) ss = 0.9 / sqrt(sqrt(sqrt(esttol))); //pow(esttol, 0.125);
        if (hfaild) ss = Min(1.0, ss);
        RKFh = CopySign(Max(hmin,ss*fabs(RKFh)), RKFh);
        
// Yang: step recording to the resfile BinaryDance.bt------------
        
        FixNorm(y);
        resfile<<setw(WidthDouble)<<t;
        for(i=0; i<dim; i++) resfile<<setw(WidthDouble)<<y[i];
        resfile<<endl;
        
        testConserv(y, alpha, beta);

        //cout<<"2"<<endl;
        
//---------------------------------------------------------------
        
        if (output) 
		{
            t = tout;
			RKFiflag = 2;		
			goto final;
        }

        if (RKFiflag <= 0)
		{ // one-step mode
            RKFiflag = -2;
            goto final;
        }
    }
final:
    
    cout<<"t = "<<t<<endl;
    
// Yang: close the resfile BinaryDance.bt------
    
    resfile.close();
//---------------------------------------------
    
	for(i=0;i<14;i++) delete[] RKFWork[i];
	delete[] RKFWork;
	return;
}
}//end namespace RKF78

//fix the normality of the orientation quaternions
//Alpha: y6, y7, y8, y9; Beta: y19, y20, y21, y22
//
void FixNorm(double* y)
{
    int i;
    double tmpd1, tmpd2;
    
    tmpd1 = 0.0;
    tmpd2 = 0.0;
    
    for (i=0; i<4; i++)
    {
        tmpd1 = tmpd1 + y[i+6]*y[i+6];
        tmpd2 = tmpd2 + y[i+19]*y[i+19];
    }
    
    assert(tmpd1 > 0.0);
    tmpd1 = sqrt(tmpd1);
    assert(tmpd2 > 0.0);
    tmpd2 = sqrt(tmpd2);
    
    for (i=0; i<4; i++)
    {
        y[i+6] = y[i+6] / tmpd1;
        y[i+19] = y[i+19] / tmpd2;;
    }
}


