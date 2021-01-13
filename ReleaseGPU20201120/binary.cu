//
//  binary.cpp
//  
//
//  Created by Yang Yu & Yunfeng Gao on 10/15/20.
//  The reduction algorithm here refers to the code of Hui Liu, https://github.com/huiscliu/tutorials
//

#include "binary.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "reduction_aux.h"

#include <stdio.h>
#include <cuda.h>
#include <math.h>


#include <sys/time.h>
#include <time.h>

#define TX 16
#define TY 16
#define N1 6027
#define N2 4980
#define TNUM 1




__device__
void vectorCross1( double* v1, double* v2,double* v)
{
    double a, b, c;
    
    a = v1[1]*v2[2] - v1[2]*v2[1];
    b = v1[2]*v2[0] - v1[0]*v2[2];
    c = v1[0]*v2[1] - v1[1]*v2[0];
    
	v[0] = a;
	v[1] = b;
	v[2] = c;

}

void vectorAdd2( double* v1, double* v2,double* v)
{
	v[0] = v1[0] + v2[0];
	v[1] = v1[1] + v2[1];
	v[2] = v1[2] + v2[2];
}

void vectorSet2(double* v, double x, double y, double z)
{
	v[0] = x;
	v[1] = y;
	v[2] = z;
}

void vectorCross2( double* v1, double* v2,double* v)
{
    double a, b, c;
    
    a = v1[1]*v2[2] - v1[2]*v2[1];
    b = v1[2]*v2[0] - v1[0]*v2[2];
    c = v1[0]*v2[1] - v1[1]*v2[0];
    
	v[0] = a;
	v[1] = b;
	v[2] = c;
}



void vectorZero2(double* v)
{
v[0] = 0;
v[1] = 0;
v[2] = 0;
}


void vectorScale2( double* u,const double s,double* v)
{
	v[0] = u[0]*s;
	v[1] = u[1]*s;
	v[2] = u[2]*s;
}

__global__
void testkernel(double* d_GFAx,double* d_GFAy,double* d_GFAz, double* d_GTAx, double* d_GTAy, double* d_GTAz, 
	double* d_GTBx,double* d_GTBy,double* d_GTBz,double* dx1, double* dx2, double* dx3, double* dx4, double* dx5, double* dx6, double* dwx,
	double* dy1, double* dy2, double* dy3, double* dy4, double* dy5, double* dy6, double* dwy)
{

	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;

    const int i = r * N1 + c;
    
	const double G = 6.67384e-11;

    if(c<N1 & r<N2)
    {

	double dd = sqrt((dy1[r]-dx1[c]) * (dy1[r]-dx1[c])  + (dy2[r]-dx2[c]) * (dy2[r]-dx2[c]) + (dy3[r]-dx3[c]) * (dy3[r]-dx3[c]) );
   
	double dd1 = G * dwx[c] * dwy[r] / (dd * dd * dd); 
	
    double d_VV1[3] = { (dy1[r]-dx1[c]) * dd1, (dy2[r]-dx2[c]) * dd1, (dy3[r]-dx3[c]) * dd1 };
    
	d_GFAx[i] = d_VV1[0];
	d_GFAy[i] = d_VV1[1];
	d_GFAz[i] = d_VV1[2];

	double d_VV2[3] = { dx4[c], dx5[c], dx6[c] };

	double d_VV4[3] = { 0,0,0 };
	vectorCross1(d_VV2, d_VV1, d_VV4); 

	d_GTAx[i] = d_VV4[0];
	d_GTAy[i] = d_VV4[1];
	d_GTAz[i] = d_VV4[2];


	double d_VV8[3] = { dy4[r], dy5[r], dy6[r] };

	double d_VV5[3] = { 0,0,0 };
 	vectorCross1(d_VV1, d_VV8, d_VV5);

	d_GTBx[i] = d_VV5[0];
	d_GTBy[i] = d_VV5[1];
    d_GTBz[i] = d_VV5[2];
    }
}

__device__ void warpReduce(volatile double *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void asum_stg_1(const double *x1, double *y1,
	const double *x2, double *y2,
	const double *x3, double *y3,
	const double *x4, double *y4,
	const double *x5, double *y5,
	const double *x6, double *y6,
	const double *x7, double *y7,
	const double *x8, double *y8,
	const double *x9, double *y9, int N)
{
	__shared__ double sdata1[256];
	__shared__ double sdata2[256];
	__shared__ double sdata3[256];
	__shared__ double sdata4[256];
	__shared__ double sdata5[256];
	__shared__ double sdata6[256];
	__shared__ double sdata7[256];
	__shared__ double sdata8[256];
	__shared__ double sdata9[256];
    int idx = get_tid();
    int tid = threadIdx.x;
    int bid = get_bid();

    if (idx < N) {
		sdata1[tid] = x1[idx];
		sdata2[tid] = x2[idx];
		sdata3[tid] = x3[idx];
		sdata4[tid] = x4[idx];
		sdata5[tid] = x5[idx];
		sdata6[tid] = x6[idx];
		sdata7[tid] = x7[idx];
		sdata8[tid] = x8[idx];
		sdata9[tid] = x9[idx];
    }
    else {
		sdata1[tid] = 0;
		sdata2[tid] = 0;
		sdata3[tid] = 0;
		sdata4[tid] = 0;
		sdata5[tid] = 0;
		sdata6[tid] = 0;
		sdata7[tid] = 0;
		sdata8[tid] = 0;
		sdata9[tid] = 0;
    }

    __syncthreads();

	if (tid < 128) 
	sdata1[tid] += sdata1[tid + 128],
	sdata2[tid] += sdata2[tid + 128],
	sdata3[tid] += sdata3[tid + 128],
	sdata4[tid] += sdata4[tid + 128],
	sdata5[tid] += sdata5[tid + 128],
	sdata6[tid] += sdata6[tid + 128],
	sdata7[tid] += sdata7[tid + 128],
	sdata8[tid] += sdata8[tid + 128],
	sdata9[tid] += sdata9[tid + 128];
    __syncthreads();

	if (tid < 64) 
	sdata1[tid] += sdata1[tid + 64],
	sdata2[tid] += sdata2[tid + 64],
	sdata3[tid] += sdata3[tid + 64],
	sdata4[tid] += sdata4[tid + 64],
	sdata5[tid] += sdata5[tid + 64],
	sdata6[tid] += sdata6[tid + 64],
	sdata7[tid] += sdata7[tid + 64],
	sdata8[tid] += sdata8[tid + 64],
	sdata9[tid] += sdata9[tid + 64];
    __syncthreads();

	if (tid < 32) 
	warpReduce(sdata1, tid),
	warpReduce(sdata2, tid),
	warpReduce(sdata3, tid),
	warpReduce(sdata4, tid),
	warpReduce(sdata5, tid),
	warpReduce(sdata6, tid),
	warpReduce(sdata7, tid),
	warpReduce(sdata8, tid),
	warpReduce(sdata9, tid);

	if (tid == 0) 
	y1[bid] = sdata1[0],
	y2[bid] = sdata2[0],
	y3[bid] = sdata3[0],
	y4[bid] = sdata4[0],
	y5[bid] = sdata5[0],
	y6[bid] = sdata6[0],
	y7[bid] = sdata7[0],
	y8[bid] = sdata8[0],
	y9[bid] = sdata9[0];
}

__global__ void asum_stg_3(double *x1, double *x2, double *x3, double *x4, double *x5, double *x6, double *x7, double *x8, double *x9, int N)
{
	__shared__ double sdata1[128];
	__shared__ double sdata2[128];
	__shared__ double sdata3[128];
	__shared__ double sdata4[128];
	__shared__ double sdata5[128];
	__shared__ double sdata6[128];
	__shared__ double sdata7[128];
	__shared__ double sdata8[128];
	__shared__ double sdata9[128];
    int tid = threadIdx.x;
    int i;

	sdata1[tid] = 0;
	sdata2[tid] = 0;
	sdata3[tid] = 0;
	sdata4[tid] = 0;
	sdata5[tid] = 0;
	sdata6[tid] = 0;
	sdata7[tid] = 0;
	sdata8[tid] = 0;
	sdata9[tid] = 0;
	

    for (i = 0; i < N; i += 128) {
		if (tid + i < N) sdata1[tid] += x1[i + tid];
		if (tid + i < N) sdata2[tid] += x2[i + tid];
		if (tid + i < N) sdata3[tid] += x3[i + tid];
		if (tid + i < N) sdata4[tid] += x4[i + tid];
		if (tid + i < N) sdata5[tid] += x5[i + tid];
		if (tid + i < N) sdata6[tid] += x6[i + tid];
		if (tid + i < N) sdata7[tid] += x7[i + tid];
		if (tid + i < N) sdata8[tid] += x8[i + tid];
		if (tid + i < N) sdata9[tid] += x9[i + tid];
    }

    __syncthreads();

	if (tid < 64) 
	sdata1[tid] = sdata1[tid] + sdata1[tid + 64],
	sdata2[tid] = sdata2[tid] + sdata2[tid + 64],
	sdata3[tid] = sdata3[tid] + sdata3[tid + 64],
	sdata4[tid] = sdata4[tid] + sdata4[tid + 64],
	sdata5[tid] = sdata5[tid] + sdata5[tid + 64],
	sdata6[tid] = sdata6[tid] + sdata6[tid + 64],
	sdata7[tid] = sdata7[tid] + sdata7[tid + 64],
	sdata8[tid] = sdata8[tid] + sdata8[tid + 64],
	sdata9[tid] = sdata9[tid] + sdata9[tid + 64];
    __syncthreads();

	if (tid < 32) 
	warpReduce(sdata1, tid),
	warpReduce(sdata2, tid),
	warpReduce(sdata3, tid),
	warpReduce(sdata4, tid),
	warpReduce(sdata5, tid),
	warpReduce(sdata6, tid),
	warpReduce(sdata7, tid),
	warpReduce(sdata8, tid),
	warpReduce(sdata9, tid);

	if (tid == 0) 
	x1[0] = sdata1[0],
	x2[0] = sdata2[0],
	x3[0] = sdata3[0],
	x4[0] = sdata4[0],
	x5[0] = sdata5[0],
	x6[0] = sdata6[0],
	x7[0] = sdata7[0],
	x8[0] = sdata8[0],
	x9[0] = sdata9[0];
}

__global__ void asum_stg_4(double *ans1, double *dz1,  double *ans2, double *dz2, double *ans3,
	 double *dz3, double *ans4, double *dz4, double *ans5, double *dz5, double *ans6, double *dz6, 
	 double *ans7, double *dz7, double *ans8, double *dz8, double *ans9, double *dz9 )
{
	int tid = threadIdx.x;
	ans1[tid] = dz1[0];
	ans2[tid] = dz2[0];
	ans3[tid] = dz3[0];
	ans4[tid] = dz4[0];
	ans5[tid] = dz5[0];
	ans6[tid] = dz6[0];
	ans7[tid] = dz7[0];
	ans8[tid] = dz8[0];
	ans9[tid] = dz9[0];
}


void asum(double* dans1, double *dx1, double *dy1, double *dz1,
	double* dans2, double *dx2, double *dy2, double *dz2,
	double* dans3, double *dx3, double *dy3, double *dz3,
	double* dans4, double *dx4, double *dy4, double *dz4,
	double* dans5, double *dx5, double *dy5, double *dz5,
	double* dans6, double *dx6, double *dy6, double *dz6,
	double* dans7, double *dx7, double *dy7, double *dz7,
	double* dans8, double *dx8, double *dy8, double *dz8,
	double* dans9, double *dx9, double *dy9, double *dz9, int N)
{

    int bs = 256;

    int s = ceil(sqrt((N + bs - 1.) / bs));
    dim3 grid = dim3(s, s);
    int gs = 0;

    asum_stg_1<<<grid, bs>>>(dx1, dy1, dx2, dy2,  dx3, dy3,  dx4, dy4,  dx5, dy5,  dx6, dy6,  dx7, dy7,  dx8, dy8,  dx9, dy9, N);

    {
        int N8 = (N + bs - 1) / bs;

        int s2 = ceil(sqrt((N8 + bs - 1.) / bs));
        dim3 grid2 = dim3(s2, s2);

        asum_stg_1<<<grid2, bs>>>(dy1, dz1, dy2, dz2,  dy3, dz3,  dy4, dz4,  dy5, dz5,  dy6, dz6,  dy7, dz7,  dy8, dz8,  dy9, dz9,  N8);

        gs = (N8 + bs - 1.) / bs;
    }

	asum_stg_3<<<1, 128>>>(dz1, dz2, dz3, dz4, dz5, dz6, dz7, dz8, dz9,gs);
	
	asum_stg_4<<<1,32>>>(dans1, dz1, dans2, dz2, dans3, dz3, dans4, dz4, dans5, dz5, dans6, dz6, dans7, dz7, dans8, dz8, dans9, dz9);
}


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


    double td;
    td = get_time(); 
    
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

    int n1bytes = N1 * sizeof(double);
	int n2bytes = N2 * sizeof(double);
	int n3bytes = N1 * N2 * sizeof(double);

	int N3 = N1*N2;
	int N4 = (N3  + 255) / 256;
	int N5 = (N4  + 255) / 256;
	int N6 = (N5  + 255) / 256;

    double* dx1 = NULL, * dx2 = NULL, * dx3 = NULL;
	double* dx4 = NULL, * dx5 = NULL, * dx6 = NULL;
	double* dy1 = NULL, * dy2 = NULL, * dy3 = NULL;
	double* dy4 = NULL, * dy5 = NULL, * dy6 = NULL;
	double* dwx = NULL, * dwy = NULL;
	double* d_GFAx = NULL,* d_GFAy= NULL, * d_GFAz= NULL;
	double* d_GTAx = NULL,* d_GTAy= NULL, * d_GTAz= NULL;
	double* d_GTBx = NULL,* d_GTBy = NULL,* d_GTBz = NULL;
	double* d1_GFAx = NULL, * d2_GFAx = NULL, * d3_GFAx = NULL;
	double*  de_GFAx = NULL;
	double* d1_GFAy = NULL, * d2_GFAy = NULL, * d3_GFAy = NULL;
	double*  de_GFAy = NULL;
	double* d1_GFAz = NULL, * d2_GFAz = NULL, * d3_GFAz = NULL;
	double*  de_GFAz = NULL;

    double* d1_GTAx = NULL, * d2_GTAx = NULL, * d3_GTAx = NULL;
	double*  de_GTAx = NULL;
	double* d1_GTAy = NULL, * d2_GTAy = NULL, * d3_GTAy = NULL;
	double*  de_GTAy = NULL;
	double* d1_GTAz = NULL, * d2_GTAz = NULL, * d3_GTAz = NULL;
	double*  de_GTAz = NULL;

	double* d1_GTBx = NULL, * d2_GTBx = NULL, * d3_GTBx = NULL;
	double*  de_GTBx = NULL;
	double* d1_GTBy = NULL, * d2_GTBy = NULL, * d3_GTBy = NULL;
	double*  de_GTBy = NULL;
	double* d1_GTBz = NULL, * d2_GTBz = NULL, * d3_GTBz = NULL;
    double*  de_GTBz = NULL;
    
    cudaMalloc((void**)& dx1, n1bytes);
	cudaMalloc((void**)& dx2, n1bytes);
	cudaMalloc((void**)& dx3, n1bytes);
	cudaMalloc((void**)& dx4, n1bytes);
	cudaMalloc((void**)& dx5, n1bytes);
	cudaMalloc((void**)& dx6, n1bytes);
	cudaMalloc((void**)& dwx, n1bytes);

	cudaMalloc((void**)& dy1, n2bytes);
	cudaMalloc((void**)& dy2, n2bytes);
	cudaMalloc((void**)& dy3, n2bytes);
	cudaMalloc((void**)& dy4, n2bytes);
	cudaMalloc((void**)& dy5, n2bytes);
	cudaMalloc((void**)& dy6, n2bytes);
	cudaMalloc((void**)& dwy, n2bytes);

	cudaMalloc((void**)& d_GFAx, n3bytes);
	cudaMalloc((void**)& d_GFAy, n3bytes);
	cudaMalloc((void**)& d_GFAz, n3bytes);
	cudaMalloc((void**)& d_GTAx, n3bytes);
	cudaMalloc((void**)& d_GTAy, n3bytes);
	cudaMalloc((void**)& d_GTAz, n3bytes);
	cudaMalloc((void**)& d_GTBx, n3bytes);
	cudaMalloc((void**)& d_GTBy, n3bytes);
	cudaMalloc((void**)& d_GTBz, n3bytes);


	cudaMalloc((void **)& d1_GFAx, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d2_GFAx, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d3_GFAx, sizeof(double) * N6);
	cudaMalloc((void **)& de_GFAx, sizeof(double) * 32);

	cudaMalloc((void **)& d1_GFAy, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d2_GFAy, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d3_GFAy, sizeof(double) * N6);
	cudaMalloc((void **)& de_GFAy, sizeof(double) * 32);

	cudaMalloc((void **)& d1_GFAz, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d2_GFAz, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d3_GFAz, sizeof(double) * N6);
	cudaMalloc((void **)& de_GFAz, sizeof(double) * 32);

	cudaMalloc((void **)& d1_GTAx, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d2_GTAx, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d3_GTAx, sizeof(double) * N6);
	cudaMalloc((void **)& de_GTAx, sizeof(double) * 32);

	cudaMalloc((void **)& d1_GTAy, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d2_GTAy, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d3_GTAy, sizeof(double) * N6);
	cudaMalloc((void **)& de_GTAy, sizeof(double) * 32);

	cudaMalloc((void **)& d1_GTAz, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d2_GTAz, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d3_GTAz, sizeof(double) * N6);
	cudaMalloc((void **)& de_GTAz, sizeof(double) * 32);

	cudaMalloc((void **)& d1_GTBx, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d2_GTBx, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d3_GTBx, sizeof(double) * N6);
	cudaMalloc((void **)& de_GTBx, sizeof(double) * 32);

	cudaMalloc((void **)& d1_GTBy, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d2_GTBy, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d3_GTBy, sizeof(double) * N6);
	cudaMalloc((void **)& de_GTBy, sizeof(double) * 32);

	cudaMalloc((void **)& d1_GTBz, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d2_GTBz, sizeof(double) *  ((N3 + 255) / 256));
	cudaMalloc((void **)& d3_GTBz, sizeof(double) * N6);
	cudaMalloc((void **)& de_GTBz, sizeof(double) * 32);
    
    rk78::RKF78(MotionEquation, y, alpha, beta, varpool, tin, tout, EqnDim, reltol, abstol, RKFiflag, 100000,
    dx1,dx2,dx3,dx4,dx5,dx6,dwx,dy1,dy2,dy3,dy4,dy5,dy6,dwy,d_GFAx,d_GFAy,d_GFAz,d_GTAx,d_GTAy,d_GTAz,d_GTBx,d_GTBy,d_GTBz,
    d1_GFAx,d2_GFAx,d3_GFAx,de_GFAx,d1_GFAy,d2_GFAy,d3_GFAy,de_GFAy,d1_GFAz,d2_GFAz,d3_GFAz,de_GFAz,
    d1_GTAx,d2_GTAx,d3_GTAx,de_GTAx,d1_GTAy,d2_GTAy,d3_GTAy,de_GTAy,d1_GTAz,d2_GTAz,d3_GTAz,de_GTAz,
    d1_GTBx,d2_GTBx,d3_GTBx,de_GTBx,d1_GTBy,d2_GTBy,d3_GTBy,de_GTBy,d1_GTBz,d2_GTBz,d3_GTBz,de_GTBz);

    cudaFree(d_GFAx);  
    cudaFree(d_GFAy);
    cudaFree(d_GFAz);
    cudaFree(d_GTAx);
    cudaFree(d_GTAx);
    cudaFree(d_GTAz);
    cudaFree(d_GTBx);
    cudaFree(d_GTBy);
    cudaFree(d_GTBz);
    cudaFree(dx1);
	cudaFree(dx2);
	cudaFree(dx3);
	cudaFree(dx4);
	cudaFree(dx5);
	cudaFree(dx6);
	cudaFree(dwx);

	cudaFree(dy1);
	cudaFree(dy2);
	cudaFree(dy3);
	cudaFree(dy4);
	cudaFree(dy5);
	cudaFree(dy6);
	cudaFree(dwy);
	
	cudaFree(d1_GFAx);
    cudaFree(d2_GFAx);  
    cudaFree(d3_GFAx);  
    cudaFree(de_GFAx);  

    cudaFree(d1_GFAy);
    cudaFree(d2_GFAy);  
    cudaFree(d3_GFAy);  
    cudaFree(de_GFAy);  

    cudaFree(d1_GFAz);
    cudaFree(d2_GFAz);  
    cudaFree(d3_GFAz);  
    cudaFree(de_GFAz);  

    cudaFree(d1_GTAx);
    cudaFree(d2_GTAx);  
    cudaFree(d3_GTAx);  
    cudaFree(de_GTAx);  

    cudaFree(d1_GTAy);
    cudaFree(d2_GTAy);  
    cudaFree(d3_GTAy);  
    cudaFree(de_GTAy);  

    cudaFree(d1_GTAz);
    cudaFree(d2_GTAz);  
    cudaFree(d3_GTAz);  
    cudaFree(de_GTAz);  

    cudaFree(d1_GTBx);
    cudaFree(d2_GTBx);  
    cudaFree(d3_GTBx);  
    cudaFree(de_GTBx);  

    cudaFree(d1_GTBy);
    cudaFree(d2_GTBy);  
    cudaFree(d3_GTBy);  
    cudaFree(de_GTBy);  

    cudaFree(d1_GTBz);
    cudaFree(d2_GTBz);  
    cudaFree(d3_GTBz);  
    cudaFree(de_GTBz);  
   
    td = get_time() - td;

    printf("gpu time is: %e\n", td);

}

void MotionEquation(double t, const double* y, double* yp, FEM &alpha, FEM &beta, MASCON &varpool, double* dx1,double* dx2,double* dx3,double* dx4,double* dx5,double* dx6,double* dwx,double* dy1,double* dy2,double* dy3,double* dy4,double* dy5,double* dy6,double* dwy,
    double* d_GFAx,double* d_GFAy,double* d_GFAz,double* d_GTAx,double* d_GTAy,double* d_GTAz,double* d_GTBx,double* d_GTBy,double* d_GTBz,
    double* d1_GFAx,double* d2_GFAx,double* d3_GFAx,double* de_GFAx,double* d1_GFAy,double* d2_GFAy,double* d3_GFAy,double* de_GFAy,double* d1_GFAz,double* d2_GFAz,double* d3_GFAz,double* de_GFAz,
    double* d1_GTAx,double* d2_GTAx,double* d3_GTAx,double* de_GTAx,double* d1_GTAy,double* d2_GTAy,double* d3_GTAy,double* de_GTAy,double* d1_GTAz,double* d2_GTAz,double* d3_GTAz,double* de_GTAz,
    double* d1_GTBx,double* d2_GTBx,double* d3_GTBx,double* de_GTBx,double* d1_GTBy,double* d2_GTBy,double* d3_GTBy,double* de_GTBy,double* d1_GTBz,double* d2_GTBz,double* d3_GTBz,double* de_GTBz)
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

    int n1bytes = N1 * sizeof(double);
	int n2bytes = N2 * sizeof(double);
	int n3bytes = N1 * N2 * sizeof(double);

	int N3 = N1*N2;
	int N4 = (N3  + 255) / 256;
	int N5 = (N4  + 255) / 256;
	int N6 = (N5  + 255) / 256;

    
	double x1[N1];
	double x2[N1];
	double x3[N1];
	double x4[N1];
	double x5[N1];
	double x6[N1];
	double wx[N1];
	double y1[N2];
	double y2[N2];
	double y3[N2];
	double y4[N2];
	double y5[N2];
	double y6[N2];
	double wy[N2];
	double eGFAx[32];
	double eGFAy[32];
	double eGFAz[32];
	double eGTAx[32];
	double eGTAy[32];
	double eGTAz[32];
	double eGTBx[32];
	double eGTBy[32];
	double eGTBz[32];

	cudaMallocHost((void**)& eGFAx,  sizeof(double) * 32);

    for (i = 0; i < N1; ++i) {
        x1[i] = varpool.alpha[i][0];
		x2[i] = varpool.alpha[i][1];
		x3[i] = varpool.alpha[i][2];
		x4[i] = varpool.alpha[i][3];
		x5[i] = varpool.alpha[i][4];
		x6[i] = varpool.alpha[i][5];
		wx[i] =alpha.NodeWeights[i];
	}

	for (j = 0; j < N2; ++j) {
        y1[j] = varpool.beta[j][0];
		y2[j] = varpool.beta[j][1];
		y3[j] = varpool.beta[j][2];
		y4[j] = varpool.beta[j][3];
		y5[j] = varpool.beta[j][4];
		y6[j] = varpool.beta[j][5];
		wy[j] = beta.NodeWeights[j];
	}

	cudaMemcpy(dx1, x1, n1bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dx2, x2, n1bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dx3, x3, n1bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dx4, x4, n1bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dx5, x5, n1bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dx6, x6, n1bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dwx, wx, n1bytes, cudaMemcpyHostToDevice);


	cudaMemcpy(dy1, y1, n2bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy2, y2, n2bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy3, y3, n2bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy4, y4, n2bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy5, y5, n2bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy6, y6, n2bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dwy, wy, n2bytes, cudaMemcpyHostToDevice);

	const dim3 blockSize(TX, TY);

	const int bx = (N1 + TX - 1) / TX;
	const int by = (N2 + TY - 1) / TY;
	const dim3 gridSize = dim3(bx, by);

	testkernel << <gridSize, blockSize >> > (d_GFAx, d_GFAy, d_GFAz, d_GTAx, d_GTAy, d_GTAz,  
	d_GTBx, d_GTBy, d_GTBz, dx1, dx2, dx3, dx4, dx5, dx6, dwx, dy1, dy2, dy3, dy4, dy5, dy6, dwy);

	cudaThreadSynchronize();

	asum(de_GFAx, d_GFAx, d1_GFAx, d2_GFAx, 
		de_GFAy, d_GFAy, d1_GFAy, d2_GFAy, 
		de_GFAz, d_GFAz, d1_GFAz, d2_GFAz,
		de_GTAx, d_GTAx, d1_GTAx, d2_GTAx, 
		de_GTAy, d_GTAy, d1_GTAy, d2_GTAy, 
		de_GTAz, d_GTAz, d1_GTAz, d2_GTAz,
		de_GTBx, d_GTBx, d1_GTBx, d2_GTBx, 
		de_GTBy, d_GTBy, d1_GTBy, d2_GTBy, 
		de_GTBz, d_GTBz, d1_GTBz, d2_GTBz, N3);

	cudaThreadSynchronize();

	cudaMemcpy(eGFAx, de_GFAx, sizeof(double)*32 , cudaMemcpyDeviceToHost);
	cudaMemcpy(eGFAy, de_GFAy, sizeof(double)*32 , cudaMemcpyDeviceToHost);
	cudaMemcpy(eGFAz, de_GFAz, sizeof(double)*32 , cudaMemcpyDeviceToHost);
	cudaMemcpy(eGTAx, de_GTAx, sizeof(double)*32 , cudaMemcpyDeviceToHost);
	cudaMemcpy(eGTAy, de_GTAy, sizeof(double)*32 , cudaMemcpyDeviceToHost);
	cudaMemcpy(eGTAz, de_GTAz, sizeof(double)*32 , cudaMemcpyDeviceToHost);
	cudaMemcpy(eGTBx, de_GTBx, sizeof(double)*32 , cudaMemcpyDeviceToHost);
	cudaMemcpy(eGTBy, de_GTBy, sizeof(double)*32 , cudaMemcpyDeviceToHost);
	cudaMemcpy(eGTBz, de_GTBz, sizeof(double)*32 , cudaMemcpyDeviceToHost);

    GravForceA[0] = eGFAx[2];
    GravForceA[1] = eGFAy[2];
    GravForceA[2] = eGFAz[0];

    GravTorqueA[0] = eGTAx[2];
    GravTorqueA[1] = eGTAy[2];
    GravTorqueA[2] = eGTAz[2];

    GravTorqueB[0] = eGTBx[2];
    GravTorqueB[1] = eGTBy[2];
    GravTorqueB[2] = eGTBz[2];

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
void RKFStep(void (*Model)(double t, const double* y, double* yp, FEM &alpha, FEM &beta, MASCON &varpool, double* dx1,double* dx2,double* dx3,double* dx4,double* dx5,double* dx6,double* dwx,double* dy1,double* dy2,double* dy3,double* dy4,double* dy5,double* dy6,double* dwy,
    double* d_GFAx,double* d_GFAy,double* d_GFAz,double* d_GTAx,double* d_GTAy,double* d_GTAz,double* d_GTBx,double* d_GTBy,double* d_GTBz,
    double* d1_GFAx,double* d2_GFAx,double* d3_GFAx,double* de_GFAx,double* d1_GFAy,double* d2_GFAy,double* d3_GFAy,double* de_GFAy,double* d1_GFAz,double* d2_GFAz,double* d3_GFAz,double* de_GFAz,
    double* d1_GTAx,double* d2_GTAx,double* d3_GTAx,double* de_GTAx,double* d1_GTAy,double* d2_GTAy,double* d3_GTAy,double* de_GTAy,double* d1_GTAz,double* d2_GTAz,double* d3_GTAz,double* de_GTAz,
    double* d1_GTBx,double* d2_GTBx,double* d3_GTBx,double* de_GTBx,double* d1_GTBy,double* d2_GTBy,double* d3_GTBy,double* de_GTBy,double* d1_GTBz,double* d2_GTBz,double* d3_GTBz,double* de_GTBz),
     const double* y, FEM &alpha, FEM &beta, MASCON &varpool, double t, double h, double* ss, int dim, double** RKFWork,
     double* dx1,double* dx2,double* dx3,double* dx4,double* dx5,double* dx6,double* dwx,double* dy1,double* dy2,double* dy3,double* dy4,double* dy5,double* dy6,double* dwy,
     double* d_GFAx,double* d_GFAy,double* d_GFAz,double* d_GTAx,double* d_GTAy,double* d_GTAz,double* d_GTBx,double* d_GTBy,double* d_GTBz,
     double* d1_GFAx,double* d2_GFAx,double* d3_GFAx,double* de_GFAx,double* d1_GFAy,double* d2_GFAy,double* d3_GFAy,double* de_GFAy,double* d1_GFAz,double* d2_GFAz,double* d3_GFAz,double* de_GFAz,
     double* d1_GTAx,double* d2_GTAx,double* d3_GTAx,double* de_GTAx,double* d1_GTAy,double* d2_GTAy,double* d3_GTAy,double* de_GTAy,double* d1_GTAz,double* d2_GTAz,double* d3_GTAz,double* de_GTAz,
     double* d1_GTBx,double* d2_GTBx,double* d3_GTBx,double* de_GTBx,double* d1_GTBy,double* d2_GTBy,double* d3_GTBy,double* de_GTBy,double* d1_GTBz,double* d2_GTBz,double* d3_GTBz,double* de_GTBz)
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
        Model(t + RKFa[j] * h, ss, RKFWork[j], alpha, beta, varpool,dx1,dx2,dx3,dx4,dx5,dx6,dwx,dy1,dy2,dy3,dy4,dy5,dy6,dwy,d_GFAx,d_GFAy,d_GFAz,d_GTAx,d_GTAy,d_GTAz,d_GTBx,d_GTBy,d_GTBz,
            d1_GFAx,d2_GFAx,d3_GFAx,de_GFAx,d1_GFAy,d2_GFAy,d3_GFAy,de_GFAy,d1_GFAz,d2_GFAz,d3_GFAz,de_GFAz,
            d1_GTAx,d2_GTAx,d3_GTAx,de_GTAx,d1_GTAy,d2_GTAy,d3_GTAy,de_GTAy,d1_GTAz,d2_GTAz,d3_GTAz,de_GTAz,
            d1_GTBx,d2_GTBx,d3_GTBx,de_GTBx,d1_GTBy,d2_GTBy,d3_GTBy,de_GTBy,d1_GTBz,d2_GTBz,d3_GTBz,de_GTBz);
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
void RKF78(void (*Model)(double t, const double* y, double* yp, FEM &alpha, FEM &beta, MASCON &varpool,
    double* dx1,double* dx2,double* dx3,double* dx4,double* dx5,double* dx6,double* dwx,double* dy1,double* dy2,double* dy3,double* dy4,double* dy5,double* dy6,double* dwy,
    double* d_GFAx,double* d_GFAy,double* d_GFAz,double* d_GTAx,double* d_GTAy,double* d_GTAz,double* d_GTBx,double* d_GTBy,double* d_GTBz,
    double* d1_GFAx,double* d2_GFAx,double* d3_GFAx,double* de_GFAx,double* d1_GFAy,double* d2_GFAy,double* d3_GFAy,double* de_GFAy,double* d1_GFAz,double* d2_GFAz,double* d3_GFAz,double* de_GFAz,
    double* d1_GTAx,double* d2_GTAx,double* d3_GTAx,double* de_GTAx,double* d1_GTAy,double* d2_GTAy,double* d3_GTAy,double* de_GTAy,double* d1_GTAz,double* d2_GTAz,double* d3_GTAz,double* de_GTAz,
    double* d1_GTBx,double* d2_GTBx,double* d3_GTBx,double* de_GTBx,double* d1_GTBy,double* d2_GTBy,double* d3_GTBy,double* de_GTBy,double* d1_GTBz,double* d2_GTBz,double* d3_GTBz,double* de_GTBz), 
    double* y, FEM &alpha, FEM &beta, MASCON &varpool, double& t, const double& tout, int dim, double RelTol, double* AbsTol, int& RKFiflag, int RKFmaxnfe,
double* dx1,double* dx2,double* dx3,double* dx4,double* dx5,double* dx6,double* dwx,double* dy1,double* dy2,double* dy3,double* dy4,double* dy5,double* dy6,double* dwy,
double* d_GFAx,double* d_GFAy,double* d_GFAz,double* d_GTAx,double* d_GTAy,double* d_GTAz,double* d_GTBx,double* d_GTBy,double* d_GTBz,
double* d1_GFAx,double* d2_GFAx,double* d3_GFAx,double* de_GFAx,double* d1_GFAy,double* d2_GFAy,double* d3_GFAy,double* de_GFAy,double* d1_GFAz,double* d2_GFAz,double* d3_GFAz,double* de_GFAz,
double* d1_GTAx,double* d2_GTAx,double* d3_GTAx,double* de_GTAx,double* d1_GTAy,double* d2_GTAy,double* d3_GTAy,double* de_GTAy,double* d1_GTAz,double* d2_GTAz,double* d3_GTAz,double* de_GTAz,
double* d1_GTBx,double* d2_GTBx,double* d3_GTBx,double* de_GTBx,double* d1_GTBy,double* d2_GTBy,double* d3_GTBy,double* de_GTBy,double* d1_GTBz,double* d2_GTBz,double* d3_GTBz,double* de_GTBz)
{
    // Get machine epsilon
 
    const double eps = 2.2204460492503131e-016,
                 u26 = 26*eps;               
    
    const double remin = 1e-15;	

    int mflag = abs(RKFiflag);
	int i, RKFnfe, RKFkop, RKFinit=0, RKFkflag=0, RKFjflag=0;
	double RKFh, RKFsavre=0.0, RKFsavae=0.0;

    ofstream resfile;

    ofstream resfile1;
    
    ofstream resfile2;

    ofstream resfile3;
    
    
    const char *outfile = "BinaryOutput.bt";
    const char *outfile1 = "convert.bt";
    const char *outfile2 = "MomentumMoment.bt";
    const char *outfile3= "Momentum.bt";
        
    resfile.open(outfile);
    resfile<<setiosflags(ios::scientific)<<setprecision(PrecDouble); // output precision and format

    resfile1.open(outfile1);
    resfile1<<setiosflags(ios::scientific)<<setprecision(PrecDouble); // output precision and format

    resfile2.open(outfile2);
    resfile2<<setiosflags(ios::scientific)<<setprecision(18); // output precision and format

    resfile3.open(outfile3);
    resfile3<<setiosflags(ios::scientific)<<setprecision(18); // output precision and format

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
        Model(t,y,RKFWork[0], alpha, beta, varpool,dx1,dx2,dx3,dx4,dx5,dx6,dwx,dy1,dy2,dy3,dy4,dy5,dy6,dwy,d_GFAx,d_GFAy,d_GFAz,d_GTAx,d_GTAy,d_GTAz,d_GTBx,d_GTBy,d_GTBz,
            d1_GFAx,d2_GFAx,d3_GFAx,de_GFAx,d1_GFAy,d2_GFAy,d3_GFAy,de_GFAy,d1_GFAz,d2_GFAz,d3_GFAz,de_GFAz,
            d1_GTAx,d2_GTAx,d3_GTAx,de_GTAx,d1_GTAy,d2_GTAy,d3_GTAy,de_GTAy,d1_GTAz,d2_GTAz,d3_GTAz,de_GTAz,
            d1_GTBx,d2_GTBx,d3_GTBx,de_GTBx,d1_GTBy,d2_GTBy,d3_GTBy,de_GTBy,d1_GTBz,d2_GTBz,d3_GTBz,de_GTBz);  // call function
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

        Model(tout,y,RKFWork[0], alpha, beta, varpool,dx1,dx2,dx3,dx4,dx5,dx6,dwx,dy1,dy2,dy3,dy4,dy5,dy6,dwy,d_GFAx,d_GFAy,d_GFAz,d_GTAx,d_GTAy,d_GTAz,d_GTBx,d_GTBy,d_GTBz,
            d1_GFAx,d2_GFAx,d3_GFAx,de_GFAx,d1_GFAy,d2_GFAy,d3_GFAy,de_GFAy,d1_GFAz,d2_GFAz,d3_GFAz,de_GFAz,
            d1_GTAx,d2_GTAx,d3_GTAx,de_GTAx,d1_GTAy,d2_GTAy,d3_GTAy,de_GTAy,d1_GTAz,d2_GTAz,d3_GTAz,de_GTAz,
            d1_GTBx,d2_GTBx,d3_GTBx,de_GTBx,d1_GTBy,d2_GTBy,d3_GTBy,de_GTBy,d1_GTBz,d2_GTBz,d3_GTBz,de_GTBz);
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
        
        RKFStep(Model, y, alpha, beta, varpool, t, RKFh, RKFWork[13], dim, RKFWork,dx1,dx2,dx3,dx4,dx5,dx6,dwx,dy1,dy2,dy3,dy4,dy5,dy6,dwy,d_GFAx,d_GFAy,d_GFAz,d_GTAx,d_GTAy,d_GTAz,d_GTBx,d_GTBy,d_GTBz,
            d1_GFAx,d2_GFAx,d3_GFAx,de_GFAx,d1_GFAy,d2_GFAy,d3_GFAy,de_GFAy,d1_GFAz,d2_GFAz,d3_GFAz,de_GFAz,
            d1_GTAx,d2_GTAx,d3_GTAx,de_GTAx,d1_GTAy,d2_GTAy,d3_GTAy,de_GTAy,d1_GTAz,d2_GTAz,d3_GTAz,de_GTAz,
            d1_GTBx,d2_GTBx,d3_GTBx,de_GTBx,d1_GTBy,d2_GTBy,d3_GTBy,de_GTBy,d1_GTBz,d2_GTBz,d3_GTBz,de_GTBz);
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

        Model(t,y,RKFWork[0], alpha, beta, varpool,dx1,dx2,dx3,dx4,dx5,dx6,dwx,dy1,dy2,dy3,dy4,dy5,dy6,dwy,d_GFAx,d_GFAy,d_GFAz,d_GTAx,d_GTAy,d_GTAz,d_GTBx,d_GTBy,d_GTBz,
            d1_GFAx,d2_GFAx,d3_GFAx,de_GFAx,d1_GFAy,d2_GFAy,d3_GFAy,de_GFAy,d1_GFAz,d2_GFAz,d3_GFAz,de_GFAz,
            d1_GTAx,d2_GTAx,d3_GTAx,de_GTAx,d1_GTAy,d2_GTAy,d3_GTAy,de_GTAy,d1_GTAz,d2_GTAz,d3_GTAz,de_GTAz,
            d1_GTBx,d2_GTBx,d3_GTBx,de_GTBx,d1_GTBy,d2_GTBy,d3_GTBy,de_GTBy,d1_GTBz,d2_GTBz,d3_GTBz,de_GTBz);
        RKFnfe++;     
        ss = 5.0;
        if (esttol > 2.565784513950347900390625E-8) ss = 0.9 / sqrt(sqrt(sqrt(esttol))); //pow(esttol, 0.125);
        if (hfaild) ss = Min(1.0, ss);
        RKFh = CopySign(Max(hmin,ss*fabs(RKFh)), RKFh);
        

        
        FixNorm(y);
        resfile<<setw(WidthDouble)<<t;
        for(i=0; i<dim; i++) resfile<<setw(WidthDouble)<<y[i];
        resfile<<endl;


        Vector MomentumMoment;
        Vector Momentum;
        testConserv(y, alpha, beta, MomentumMoment, Momentum);

        resfile1<<setw(WidthDouble)<<t;

        resfile1<<"| "<<MomentumMoment[0]<<" "<<MomentumMoment[1]<<" "<<MomentumMoment[2]<<" | "<<Momentum[0]<<" "<<Momentum[1]<<" "<<Momentum[2]<<" | "<<endl;
        resfile1<<endl;

        double aMomentumMoment = sqrt(MomentumMoment[0]*MomentumMoment[0]+MomentumMoment[1]*MomentumMoment[1]+MomentumMoment[2]*MomentumMoment[2]);
        double aMomentum = sqrt(Momentum[0]*Momentum[0]+Momentum[1]*Momentum[1]+Momentum[2]*Momentum[2]);

        // resfile2<<setw(WidthDouble)<<t;
        resfile2<<aMomentumMoment<<endl;
        // resfile2<<endl;
        // resfile3<<setw(WidthDouble)<<t;
        resfile3<<aMomentum<<endl;
        // resfile3<<endl;

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