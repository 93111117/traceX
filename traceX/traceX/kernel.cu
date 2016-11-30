
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#ifndef dot
#define dot(ax,ay,bx,by) (ax*bx+ay*by)
#endif



cudaError_t traceWithCuda(double *Rx, double *Ry, double *Vx, double *Vy, double *pwr, double *bndlenght,double *bndref, const unsigned int rcnt,const unsigned int bcnt,const unsigned int kbcnt);


__global__ void traceKernel(double *Rx, double *Ry, double *Vx, double *Vy, double *pwr, double *bndlenght, double *bndref, const unsigned int *sizes)
{
	int i = threadIdx.x;
	unsigned int rcnt = sizes[0], bcnt = sizes[1], kbcnt = sizes[2];
	while (true) {
		bool coll = false;
		for (int j = 0; j < bcnt; j++) {
			double det = dot(Vx[i], Vy[i], -Vx[rcnt + j], Vy[rcnt + j]);
			double t_b = (dot(Vx[i], -Vy[i], (Ry[i] - Ry[rcnt + j]), (Rx[i] - Rx[rcnt + j])))/det;
			double t = (dot(Vx[rcnt+j],-Vy[rcnt+j], (Ry[i] - Ry[rcnt + j]), (Rx[i] - Rx[rcnt + j])))/det;
			if (t > 0.0 && t_b < bndlenght[j]) {
				coll = true;
				Rx[i] = Rx[i] + Vx[i] * t;
				Ry[i] = Ry[i] + Vy[i] * t;
				double dotpro = dot(Vx[i], Vy[i], -Vy[rcnt + 1], Vx[rcnt + 1]);
				Vx[i] = Vx[i] + 2.0*dotpro*Vy[rcnt + 1];
				Vy[i] = Vy[i] - 2.0*dotpro*Vx[rcnt + 1];
				pwr[i] = pwr[i] * bndref[j];
				break;
			}
		}
		if (!coll) {
			break;
		}
	}
	
}

int main()
{
 
	const double eps = 0.000000001;
	const double d = 0.5;
	const double tgTheta = 3.0;
	const unsigned int raycount = 10;
	const unsigned int bndcount = 2;
	const unsigned int kbndcount = 1;
	double Rx[raycount + bndcount + kbndcount];
	double Ry[raycount + bndcount + kbndcount];
	double Vx[raycount + bndcount + kbndcount];
	double Vy[raycount + bndcount + kbndcount];
	double bndlenght[bndcount];
	double pwr[raycount];
	double bndref[bndcount];
	//Initialize the setup. (2*d x d*tgTheta) Triangle.
	for (int i = 0; i < raycount; i++) {
		Rx[i] = (((double)i) / ((double)raycount)*2.0*d)+eps;
		Ry[i] = 0.0;
		Vx[i] = 0.0;
		Vy[i] = 1.0;
		pwr[i] = 1.0;
	}
	for (int i = 0; i < bndcount; i++) {
		bndref[i] = 0.1;
	}
	Rx[raycount] = 0.0;
	Ry[raycount] = 0.0;
	Vx[raycount] = 1.0/hypotf(1.0,tgTheta);
	Vy[raycount] = Vx[raycount]*tgTheta;
	Rx[raycount+1] = 2.0*d;
	Ry[raycount+1] = 0.0;
	Vx[raycount+1] = -1.0 / hypotf(1.0, tgTheta);
	Vy[raycount+1] = Vx[raycount] * tgTheta;
	Rx[raycount + 2] = 0.0;
	Ry[raycount + 2] = 0.0;
	Vx[raycount + 2] = 1.0;
	Vy[raycount + 2] = 0.0;
	bndlenght[0] = d*hypot(1.0, tgTheta);
	bndlenght[1]= d*hypot(1.0, tgTheta);
    // Add vectors in parallel.
    cudaError_t cudaStatus = traceWithCuda(Rx,Ry,Vx,Vy,pwr,bndlenght,bndref,raycount,bndcount,kbndcount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	for (int i = 0; i < raycount; i++) {
		
			std::cout << pwr[i]<<std::endl;
		
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	std::getchar();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.

cudaError_t traceWithCuda(double *Rx, double *Ry, double *Vx, double *Vy, double *pwr, double *bndlenght, double *bndref, const unsigned int rcnt, const unsigned int bcnt, const unsigned int kbcnt)
{
	double *dev_Rx = 0;
	double *dev_Ry = 0;
	double *dev_Vx = 0;
	double *dev_Vy = 0;
	double *dev_pwr = 0;
	double *dev_bndlenght = 0;
	double *dev_bndref = 0;
	unsigned int *dev_sizes = 0;
	unsigned int sizes[3] = {rcnt,bcnt,kbcnt};
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for data    
	cudaStatus = cudaMalloc((void**)&dev_sizes, (3) * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_Rx, (rcnt+bcnt+kbcnt) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Ry, (rcnt + bcnt + kbcnt) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_Vx, (rcnt + bcnt + kbcnt) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_Vy, (rcnt + bcnt + kbcnt) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_pwr, (rcnt) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_bndlenght, (bcnt) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_bndref, (bcnt) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_sizes,sizes, (3) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_Rx, Rx, (rcnt+bcnt+kbcnt) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Ry, Ry, (rcnt + bcnt + kbcnt) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_Vx, Vx, (rcnt + bcnt + kbcnt) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Vy, Vy, (rcnt + bcnt + kbcnt) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_pwr, pwr, (rcnt ) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_bndlenght, bndlenght, (bcnt)* sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_bndref, bndref, (bcnt)* sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// Launch a kernel on the GPU with one thread for each element.
	traceKernel << <1, rcnt >> >(dev_Rx,dev_Ry,dev_Vx,dev_Vy,dev_pwr,dev_bndlenght,dev_bndref,dev_sizes);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "traceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Rx, dev_Rx, (rcnt+bcnt+kbcnt) * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(Ry, dev_Ry, (rcnt + bcnt + kbcnt) * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(Vx, dev_Vx, (rcnt + bcnt + kbcnt) * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(Vy, dev_Vy, (rcnt + bcnt + kbcnt) * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(pwr, dev_pwr, (rcnt) * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(dev_Rx);
	cudaFree(dev_Ry);
	cudaFree(dev_Vx);
	cudaFree(dev_Vy);
	cudaFree(dev_pwr);
	cudaFree(dev_bndlenght);
	cudaFree(dev_bndref);
	return cudaStatus;
}