/************************************************************************************
* Author: Tao Rui
* 版本: V1.0 单卡，Linux版
* 说明:
*		计算第二部分的并行。
************************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "unistd.h"
#include "global_variables.cpp"
#include <unistd.h>  //linux
//#include <direct.h>  //windows


__global__ void print_dev_matrix(float *A, int i, int j, int k, int xdim, int ydim, int zdim)
{
	int	idx = i * ydim*zdim + j * zdim + k;
	printf("dev_Matrix[%d][%d][%d] = %8f\n", i, j, k, A[idx]);
}
/************************************************************************************
* GPU计算单个矩阵的函数
************************************************************************************/
dim3 gridUHyz(npml, nx - 1); 
dim3 blockUHyz(nz);
__global__ void gpu_UHyz(float *UHyz, float *RBHyz, float *RAHyz, float *Ez)
{
	/*
	in0 UHyz  nx+1 ny     nz
	in1 RBHyz nx-1 2*npml nz
	in2 RAHyz nx-1 2*npml nz
	in3 Ez    nx+1  ny+1  nz
	UHyz = UHyz * RBHyz + RAHyz * (Ez - Ez) / dy
	运算块大小 nx-1 * npml * nz
	UHyz由5个矩阵相乘或相加得来。
	y维分为了两块

	UHyz(2:nx, [1:npml ny-npml+1:ny], :)=RBHyz .* UHyz(2:nx, [1:npml ny-npml+1:ny], :)...
	+RAHyz ./ dy .* (Ez(2:nx, [2:npml+1 ny-npml+2:ny+1], :) - Ez(2:nx, [1:npml ny-npml+1:ny], :));
	*/

	int ix = blockIdx.y;   // ix in [0, nx - 1)
	int iy = blockIdx.x;   // iy in [0, npml)
	int iz = threadIdx.x;  // iz in [0, nz)

	int lid0 = (ix + 1)*ny*nz + iy * nz + iz; // checked!
	int rid0 = (ix + 1)*ny*nz + (iy + ny - npml) * nz + iz;  //checked!

	int lid1 = ix * (2 * npml)*nz + iy * nz + iz; // checked!
	int rid1 = ix * (2 * npml)*nz + (iy + npml) * nz + iz; // checked!

	int lid2 = lid1; // checked!
	int rid2 = rid1; // checked!

	int lid3 = (ix + 1)*(ny + 1)*nz + (iy + 1)*nz + iz; // checked!
	int rid3 = (ix + 1)*(ny + 1)*nz + (iy + ny - npml + 1)*nz + iz; // checked!

	int lid4 = (ix + 1)*(ny + 1)*nz + iy * nz + iz; // checked!
	int rid4 = (ix + 1)*(ny + 1)*nz + (iy + ny - npml)*nz + iz; // checked!

	UHyz[lid0] = UHyz[lid0] * RBHyz[lid1] + RAHyz[lid2] * (Ez[lid3] - Ez[lid4]) / dy;
	UHyz[rid0] = UHyz[rid0] * RBHyz[rid1] + RAHyz[rid2] * (Ez[rid3] - Ez[rid4]) / dy;
}
dim3 gridUHzy(nx - 1, ny);
dim3 blockUHzy(npml);
__global__ void gpu_UHzy(float *UHzy, float *RBHzy, float *RAHzy, float *Ey)
{
	/*
	in0 UHzy  --size--  nx+1  ny  nz
	in1 RBHzy --size--  nx-1  ny  2*npml
	in2 RAHzy --size--  nx-1  ny  2*npml
	in3 Ey    --size--  nx+1  ny  nz+1
	UHyz = UHyz * RBHyz + RAHyz * (Ez - Ez) / dy
	运算块大小 nx-1 * ny * (5 *npml)
	UHyz由5个矩阵相乘或相加得来。
	z维分为了两块
	UHzy(2:nx, :, [1:npml nz-npml+1:nz])=RBHzy.*UHzy(2:nx, :, [1:npml nz-npml+1:nz])
	+RAHzy./dz.*(Ey(2:nx, :, [2:npml+1 nz-npml+2:nz+1])-Ey(2:nx, :, [1:npml nz-npml+1:nz]));
	*/

	int ix = blockIdx.x;  // ix in [0, nx - 1)
	int iy = blockIdx.y;  // iy in [0, ny)
	int iz = threadIdx.x; // ix in [0, npml)

	int lid0 = (ix + 1) * ny * nz + iy * nz + iz; //checked!
	int rid0 = (ix + 1) * ny * nz + iy * nz + iz + nz - npml; //checked!

	int lid1 = ix * ny * (2 * npml) + iy * (2 * npml) + iz; //checked!
	int rid1 = ix * ny * (2 * npml) + iy * (2 * npml) + iz + npml; //checked!

	int lid2 = lid1;
	int rid2 = rid1;

	int lid4 = (ix + 1) * ny * (nz + 1) + iy * (nz + 1) + iz; //checked!
	int rid4 = (ix + 1) * ny * (nz + 1) + iy * (nz + 1) + iz + nz - npml; //checked!

	int lid3 = lid4 + 1;
	int rid3 = rid4 + 1;

	UHzy[lid0] = UHzy[lid0] * RBHzy[lid1] + RAHzy[lid2] * (Ey[lid3] - Ey[lid4]) / dz;
	UHzy[rid0] = UHzy[rid0] * RBHzy[rid1] + RAHzy[rid2] * (Ey[rid3] - Ey[rid4]) / dz;
}
dim3 gridUHzx(nx, ny - 1);
dim3 blockUHzx(npml);
__global__ void gpu_UHzx(float *UHzx, float *RBHzx, float *RAHzx, float *Ex)
{
	/*
	in0 UHzx  --size--  nx   ny + 1  nz
	in1 RBHzx --size--  nx   ny - 1  2 * npml
	in2 RAHzx --size--  nx   ny - 1  2 * npml
	in3 Ex    --size--  nx   ny + 1  nz + 1
	UHzx = UHzx * RBHzx + RAHzx * (Ez - Ez) / dy
	运算块大小 nx * ny - 1 * npml
	UHzx由5个矩阵相乘或相加得来。
	z维分为了两块  1:npml    -npml:0
	UHzx(:, 2:ny, [1:npml nz - npml + 1:nz])=RBHzx. * UHzx(:, 2:ny, [1:npml nz - npml + 1:nz])
	+RAHzx./dz.*(Ex(:, 2:ny, [2:npml + 1 nz - npml + 2:nz + 1]) - Ex(:, 2:ny, [1:npml nz - npml + 1:nz]));
	*/

	int ix = blockIdx.x;  // ix in [0, nx)
	int iy = blockIdx.y;  // iy in [0, ny - 1)
	int iz = threadIdx.x; // iz in [0, npml)

	int lid0 = ix * (ny + 1) * nz + (iy + 1) * nz + iz; // checked!
	int rid0 = ix * (ny + 1) * nz + (iy + 1) * nz + iz + nz - npml; // checked!

	int lid1 = ix * (ny - 1) * (2 * npml) + iy * (2 * npml) + iz; // checked!
	int rid1 = ix * (ny - 1) * (2 * npml) + iy * (2 * npml) + iz + npml; // checked!

	int lid2 = lid1;
	int rid2 = rid1;

	int lid4 = ix * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + iz; // checked!
	int rid4 = ix * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + iz + nz - npml; // checked!

	int lid3 = lid4 + 1;
	int rid3 = rid4 + 1;

	UHzx[lid0] = UHzx[lid0] * RBHzx[lid1] + RAHzx[lid2] * (Ex[lid3] - Ex[lid4]) / dz;
	UHzx[rid0] = UHzx[rid0] * RBHzx[rid1] + RAHzx[rid2] * (Ex[rid3] - Ex[rid4]) / dz;
}
dim3 gridUHxz(npml, ny - 1);
dim3 blockUHxz(nz);
__global__ void gpu_UHxz(float *UHxz, float *RBHxz, float *RAHxz, float *Ez)
{
	/*
	in0 UHxz  --size--  nx       ny + 1  nz
	in1 RBHxz --size--  2*npml   ny - 1  nz
	in2 RAHxz --size--  2*npml   ny - 1  nz
	in3 Ez    --size--  nx + 1   ny + 1  nz
	UHxz = UHxz * RBHxz + RAHxz * (Ez - Ez) / dx
	运算块大小 npml * ny - 1 * nz
	UHxz由5个矩阵相乘或相加得来。
	x维分为了两块  1:npml    -npml:0
	UHxz([1:npml nx-npml+1:nx], 2:ny, :)=RBHxz.*UHxz([1:npml nx-npml+1:nx], 2:ny, :)...
	+RAHxz./dx.*(Ez([2:npml+1 nx-npml+2:nx+1], 2:ny, :)-Ez([1:npml nx-npml+1:nx], 2:ny, :));
	*/
	int ix = blockIdx.x;  // ix in [0, npml)
	int iy = blockIdx.y;  // iy in [0, ny - 1)
	int iz = threadIdx.x; // iz in [0, nz)

	int lid0 = ix * (ny + 1) * nz + (iy + 1) * nz + iz; // checked!
	int rid0 = (ix + nx - npml) * (ny + 1) * nz + (iy + 1) * nz + iz; // checked!

	int lid1 = ix * (ny - 1) * nz + iy * nz + iz; // checked!
	int rid1 = (ix + npml) * (ny - 1) * nz + iy * nz + iz; // checked!

	int lid2 = lid1;
	int rid2 = rid1;

	int lid4 = ix * (ny + 1) * nz + (iy + 1) * nz + iz; // checked!
	int rid4 = (ix + nx - npml) * (ny + 1) * nz + (iy + 1) * nz + iz; // checked!

	int lid3 = lid4 + (ny + 1) * nz;
	int rid3 = rid4 + (ny + 1) * nz;

	UHxz[lid0] = UHxz[lid0] * RBHxz[lid1] + RAHxz[lid2] * (Ez[lid3] - Ez[lid4]) / dx;
	UHxz[rid0] = UHxz[rid0] * RBHxz[rid1] + RAHxz[rid2] * (Ez[rid3] - Ez[rid4]) / dx;
}
dim3 gridUHxy(npml, ny);
dim3 blockUHxy(nz - 1);
__global__ void gpu_UHxy(float *UHxy, float *RBHxy, float *RAHxy, float *Ey)
{
	/*
	in0 UHxy  --size--  nx       ny      nz + 1
	in1 RBHxy --size--  2*npml   ny      nz - 1
	in2 RAHxy --size--  2*npml   ny      nz - 1
	in3 EY    --size--  nx + 1   ny      nz + 1
	UHxy = UHxy * RBHxy + RAHxy * (Ez - Ez) / dx
	运算块大小 npml * ny * nz - 1
	UHxy由5个矩阵相乘或相加得来。
	x维分为了两块  1:npml    -npml:0
	UHxy([1:npml nx-npml+1:nx], :, 2:nz)=RBHxy.*UHxy([1:npml nx-npml+1:nx], :, 2:nz)...
	+RAHxy./dx.*(Ey([2:npml+1 nx-npml+2:nx+1], :, 2:nz)-Ey([1:npml nx-npml+1:nx], :, 2:nz));
	*/
	int ix = blockIdx.x;  // ix in [0, npml)
	int iy = blockIdx.y;  // iy in [0, ny)
	int iz = threadIdx.x; // iz in [0, nz - 1)

	int lid0 = ix * ny * (nz + 1) + iy * (nz + 1) + iz + 1; // checked!
	int rid0 = (ix + nx - npml) * ny * (nz + 1) + iy * (nz + 1) + iz + 1; //checked

	int lid1 = ix * ny * (nz - 1) + iy * (nz - 1) + iz; // checked!
	int rid1 = (ix + npml) * ny * (nz - 1) + iy * (nz - 1) + iz; // checked!

	int lid2 = lid1;
	int rid2 = rid1;

	int lid4 = ix * ny * (nz + 1) + iy * (nz + 1) + iz + 1; // checked!
	int rid4 = (ix + nx - npml) * ny * (nz + 1) + iy * (nz + 1) + iz + 1; // checked!

	int lid3 = lid4 + ny * (nz + 1);
	int rid3 = rid4 + ny * (nz + 1);

	UHxy[lid0] = UHxy[lid0] * RBHxy[lid1] + RAHxy[lid2] * (Ey[lid3] - Ey[lid4]) / dx;
	UHxy[rid0] = UHxy[rid0] * RBHxy[rid1] + RAHxy[rid2] * (Ey[rid3] - Ey[rid4]) / dx;
}
dim3 gridUHyx(npml, nx);
dim3 blockUHyx(nz - 1);
__global__ void gpu_UHyx(float *UHyx, float *RBHyx, float *RAHyx, float *Ex)
{
	/*
	in0 UHyx  nx   ny     nz + 1
	in1 RBHyx nx   2*npml nz - 1
	in2 RAHyx nx   2*npml nz - 1
	in3 Ex    nx   ny + 1 nz + 1
	UHyx = UHyx * RBHyx + RAHyx * (Ex - Ex) / dy
	运算块大小 nx * npml * nz - 1
	UHyx由5个矩阵相乘或相加得来。
	y维分为了两块

	UHyx(:, [1:npml ny-npml+1:ny], 2:nz)=RBHyx.*UHyx(:, [1:npml ny-npml+1:ny], 2:nz)...
	+RAHyx./dy.*(Ex(:, [2:npml+1 ny-npml+2:ny+1], 2:nz)-Ex(:, [1:npml ny-npml+1:ny], 2:nz));
	*/
	int ix = blockIdx.y;   // ix in [0, nx)
	int iy = blockIdx.x;   // iy in [0, npml)
	int iz = threadIdx.x;  // iz in [0, nz - 1)

	int lid0 = ix * ny * (nz + 1) + iy * (nz + 1) + iz + 1; // checked!
	int rid0 = ix * ny * (nz + 1) + (iy + ny - npml) * (nz + 1) + iz + 1;  //checked!

	int lid1 = ix * (2 * npml) * (nz - 1) + iy * (nz - 1) + iz; // checked!
	int rid1 = ix * (2 * npml) * (nz - 1) + (iy + npml) * (nz - 1) + iz; // checked!

	int lid2 = lid1; // checked!
	int rid2 = rid1; // checked!


	int lid4 = ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz + 1; // checked!
	int rid4 = ix * (ny + 1) * (nz + 1) + (iy + ny - npml) * (nz + 1) + iz + 1; // checked!


	int lid3 = lid4 + (nz + 1); // checked!
	int rid3 = rid4 + (nz + 1); // checked!

	UHyx[lid0] = UHyx[lid0] * RBHyx[lid1] + RAHyx[lid2] * (Ex[lid3] - Ex[lid4]) / dy;
	UHyx[rid0] = UHyx[rid0] * RBHyx[rid1] + RAHyx[rid2] * (Ex[rid3] - Ex[rid4]) / dy;
}
dim3 gridHx(nx - 1, ny);
dim3 blockHx(nz);
__global__ void gpu_Hx(float *Hx, float *CPHx, float *CQHx, float *ky_Hx, float *kz_Hx, float *Ez, float *Ey, float *UHyz, float *UHzy)
{
	//
	// * 运算块大小 nx - 1 * ny * nz
	// * Hx(2:nx,:,:)
	//
	int ix = blockIdx.x + 1;
	int iy = blockIdx.y;
	int iz = threadIdx.x;

	int idx = ix * ny * nz + iy * nz + iz;
	int idxEz = ix * (ny + 1)*nz + iy * nz + iz;
	int idxEy = ix * ny * (nz + 1) + iy * (nz + 1) + iz;

	int deltaEz = nz;
	int deltaEy = 1;
	float CQH = CQHx[idx];

	Hx[idx] = Hx[idx] * CPHx[idx]
		- CQH / ky_Hx[idx] * (Ez[idxEz + deltaEz] - Ez[idxEz]) / dy
		+ CQH / kz_Hx[idx] * (Ey[idxEy + deltaEy] - Ey[idxEy]) / dz
		- CQH * UHyz[idx]
		+ CQH * UHzy[idx];
}
dim3 gridHy(nx, ny - 1);
dim3 blockHy(nz);
__global__ void gpu_Hy(float *Hy, float *CPHy, float *CQHy, float *kz_Hy, float *kx_Hy, float *Ex, float *Ez, float *UHzx, float *UHxz)
{
	//
	// * 运算块大小 nx * ny -1 * nz
	// * Hy(:,2:ny,:)
	//
	int ix = blockIdx.x;
	int iy = blockIdx.y + 1;
	int iz = threadIdx.x;

	int idx = ix * (ny + 1)*nz + iy * nz + iz;
	int idxEx = ix * (ny + 1)*(nz + 1) + iy * (nz + 1) + iz;
	int idxEz = ix * (ny + 1)*nz + iy * nz + iz;

	int deltaEx = 1;
	int deltaEz = (ny + 1)*nz;
	float CQH = CQHy[idx];

	Hy[idx] = Hy[idx] * CPHy[idx]
		- CQH / kz_Hy[idx] * (Ex[idxEx + deltaEx] - Ex[idxEx]) / dz
		+ CQH / kx_Hy[idx] * (Ez[idxEz + deltaEz] - Ez[idxEz]) / dx
		- CQH * UHzx[idx]
		+ CQH * UHxz[idx];
}
dim3 gridHz(nx, ny);
dim3 blockHz(nz - 1);
__global__ void gpu_Hz(float *Hz, float *CPHz, float *CQHz, float *kx_Hz, float *ky_Hz, float *Ey, float *Ex, float *UHxy, float *UHyx)
{
	//
	// * 运算块大小 nx * ny * nz -1
	// * Hz(:,;,2:nz)
	// * Hz大小为nx ny nz+1
	//
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = threadIdx.x + 1;

	int idx = ix * ny * (nz + 1) + iy * (nz + 1) + iz;
	int idxEy = ix * ny * (nz + 1) + iy * (nz + 1) + iz;
	int idxEx = ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz;
	int deltaEy = ny * (nz + 1);
	int deltaEx = nz + 1;
	float CQH = CQHz[idx];

	Hz[idx] = Hz[idx] * CPHz[idx]
		- CQH / kx_Hz[idx] * (Ey[idxEy + deltaEy] - Ey[idxEy]) / dx
		+ CQH / ky_Hz[idx] * (Ex[idxEx + deltaEx] - Ex[idxEx]) / dy
		- CQH * UHxy[idx]
		+ CQH * UHyx[idx];
}
dim3 gridUEyz(npml - 1, nx);
dim3 blockUEyz(nz - 1);
__global__ void gpu_UEyz(float *UEyz, float *RBEyz, float *RAEyz, float *Hz)
{
	/*
	dim3 blockUEyz(nz - 1);
	dim3 gridUEyz(npml - 1, nx);

	in0 UEyz  nx   ny + 1     nz + 1
	in1 RBEyz nx   2*(npml-1) nz - 1
	in2 RAEyz nx   2*(npml-1) nz - 1
	in3 Hz    nx   ny         nz + 1

	运算块大小 nx * npml - 1 * nz - 1

	UEyz(:, [2:npml ny-npml+2:ny], 2:nz)=RBEyz .* UEyz(:, [2:npml ny-npml+2:ny], 2:nz)...
	+RAEyz ./ dy .* (Hz(:, [2:npml ny-npml+2:ny], 2:nz) - Hz(:, [1:npml-1 ny-npml+1:ny-1], 2:nz));
	*/
	int ix = blockIdx.y;   // ix in [0, nx)
	int iy = blockIdx.x;   // iy in [0, npml - 1)
	int iz = threadIdx.x;  // iz in [0, nz - 1)

	int lid0 = ix * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + (iz + 1); // checked!
	int rid0 = ix * (ny + 1) * (nz + 1) + (iy + 1 + ny - npml) * (nz + 1) + (iz + 1);  //checked!

	int lid1 = ix * (2 * (npml - 1)) * (nz - 1) + iy * (nz - 1) + iz; // checked!
	int rid1 = ix * (2 * (npml - 1)) * (nz - 1) + (iy + npml - 1) * (nz - 1) + iz; // checked!

	int lid2 = lid1; // checked!
	int rid2 = rid1; // checked!

	int lid4 = ix * ny * (nz + 1) + iy * (nz + 1) + (iz + 1); // checked!
	int rid4 = ix * ny * (nz + 1) + (iy + ny - npml) * (nz + 1) + (iz + 1); // checked!

	int lid3 = lid4 + (nz + 1); // checked!
	int rid3 = rid4 + (nz + 1); // checked!

	UEyz[lid0] = UEyz[lid0] * RBEyz[lid1] + RAEyz[lid2] * (Hz[lid3] - Hz[lid4]) / dy;
	UEyz[rid0] = UEyz[rid0] * RBEyz[rid1] + RAEyz[rid2] * (Hz[rid3] - Hz[rid4]) / dy;
}
dim3 gridUEyx(npml - 1, nx);
dim3 blockUEyx(nz - 1);
__global__ void gpu_UEyx(float *UEyx, float *RBEyx, float *RAEyx, float *Hx)
{
	/*
	dim3 blockUEyx(nz - 1);
	dim3 gridUEyx(npml - 1, nx);

	in0 UEyx  nx + 1 ny + 1     nz
	in1 RBEyx nx - 1 2*(npml-1) nz
	in2 RAEyx nx - 1 2*(npml-1) nz
	in3 Hx    nx + 1 ny         nz

	运算块大小 nx * npml-1 * nz-1

	UEyx(2:nx, [2:npml ny-npml+2:ny], :)=RBEyx .* UEyx(2:nx, [2:npml ny-npml+2:ny], :)...
	+RAEyx ./ dy .* (Hx(2:nx, [2:npml ny-npml+2:ny], :) - Hx(2:nx, [1:npml-1 ny-npml+1:ny-1], :));
	*/
	int ix = blockIdx.y;   // ix in [0, nx)
	int iy = blockIdx.x;   // iy in [0, npml - 1)
	int iz = threadIdx.x;  // iz in [0, nz - 1)

	int lid0 = (ix + 1) * (ny + 1) * nz + (iy + 1) * nz + iz; // checked!
	int rid0 = (ix + 1) * (ny + 1) * nz + (iy + 1 + ny - npml) * nz + iz;  //checked!

	int lid1 = ix * (2 * (npml - 1)) * nz + iy * nz + iz; // checked!
	int rid1 = ix * (2 * (npml - 1)) * nz + (iy + npml - 1) * nz + iz; // checked!

	int lid2 = lid1; // checked!
	int rid2 = rid1; // checked!

	int lid4 = (ix + 1) * ny * nz + iy * nz + iz; // checked!
	int rid4 = (ix + 1) * ny * nz + (iy + ny - npml) * nz + iz; // checked!

	int lid3 = lid4 + nz; // checked!
	int rid3 = rid4 + nz; // checked!

	UEyx[lid0] = UEyx[lid0] * RBEyx[lid1] + RAEyx[lid2] * (Hx[lid3] - Hx[lid4]) / dy;
	UEyx[rid0] = UEyx[rid0] * RBEyx[rid1] + RAEyx[rid2] * (Hx[rid3] - Hx[rid4]) / dy;
}
dim3 gridUExy(npml - 1, ny - 1);
dim3 blockUExy(nz);
__global__ void gpu_UExy(float *UExy, float *RBExy, float *RAExy, float *Hy)
{
	/*
	dim3 blockUExy(nz);
	dim3 gridUExy(npml - 1, ny - 1);

	in0 UExy  nx + 1     ny + 1 nz
	in1 RBExy 2*(npml-1) ny - 1 nz
	in2 RAExy 2*(npml-1) ny - 1 nz
	in3 Hy    nx         ny + 1 nz

	运算块大小 npml-1 * ny-1 * nz

	UExy([2:npml nx-npml+2:nx], 2:ny, :)=RBExy .* UExy([2:npml nx-npml+2:nx], 2:ny, :)...
	+RAExy ./ dx .* (Hy([2:npml nx-npml+2:nx], 2:ny, :) - Hy([1:npml-1 nx-npml+1:nx-1], 2:ny, :));
	*/
	int ix = blockIdx.x;   // ix in [0, npml - 1)
	int iy = blockIdx.y;   // iy in [0, ny - 1)
	int iz = threadIdx.x;  // iz in [0, nz)

	int lid0 = (ix + 1) * (ny + 1) * nz + (iy + 1) * nz + iz; // checked!
	int rid0 = (ix + 1 + nx - npml) * (ny + 1) * nz + (iy + 1) * nz + iz;  //checked!

	int lid1 = ix * (ny - 1) * nz + iy * nz + iz; // checked!
	int rid1 = (ix + npml - 1) * (ny - 1) * nz + iy * nz + iz; // checked!

	int lid2 = lid1; // checked!
	int rid2 = rid1; // checked!

	int lid4 = ix * (ny + 1) * nz + (iy + 1) * nz + iz; // checked!
	int rid4 = (ix + nx - npml) * (ny + 1) * nz + (iy + 1) * nz + iz; // checked!

	int lid3 = lid4 + (ny + 1) * nz; // checked!
	int rid3 = rid4 + (ny + 1) * nz; // checked!

	UExy[lid0] = UExy[lid0] * RBExy[lid1] + RAExy[lid2] * (Hy[lid3] - Hy[lid4]) / dx;
	UExy[rid0] = UExy[rid0] * RBExy[rid1] + RAExy[rid2] * (Hy[rid3] - Hy[rid4]) / dx;
}
dim3 gridUExz(npml - 1, ny);
dim3 blockUExz(nz - 1);
__global__ void gpu_UExz(float *UExz, float *RBExz, float *RAExz, float *Hz)
{
	/*
	dim3 blockUExz(nz - 1);
	dim3 gridUExz(npml - 1, ny);

	in0 UExz  nx + 1     ny     nz + 1
	in1 RBExz 2*(npml-1) ny     nz - 1
	in2 RAExz 2*(npml-1) ny     nz - 1
	in3 Hz    nx         ny     nz + 1
	运算块大小 npml-1 * ny * nz-1

	UExz([2:npml nx-npml+2:nx], :, 2:nz)=RBExz .* UExz([2:npml nx-npml+2:nx], :, 2:nz)...
	+RAExz ./ dx .* (Hz([2:npml nx-npml+2:nx], :, 2:nz) - Hz([1:npml-1 nx-npml+1:nx-1], :, 2:nz));
	*/
	int ix = blockIdx.x;   // ix in [0, npml - 1)
	int iy = blockIdx.y;   // iy in [0, ny)
	int iz = threadIdx.x;  // iz in [0, nz - 1)

	int lid0 = (ix + 1) * ny * (nz + 1) + iy * (nz + 1) + (iz + 1); // checked!
	int rid0 = (ix + 1 + nx - npml) * ny * (nz + 1) + iy * (nz + 1) + (iz + 1);  //checked!

	int lid1 = ix * ny * (nz - 1) + iy * (nz - 1) + iz; // checked!
	int rid1 = (ix + npml - 1) * ny * (nz - 1) + iy * (nz - 1) + iz; // checked!

	int lid2 = lid1; // checked!
	int rid2 = rid1; // checked!

	int lid4 = ix * ny * (nz + 1) + iy * (nz + 1) + (iz + 1); // checked!
	int rid4 = (ix + nx - npml) * ny * (nz + 1) + iy * (nz + 1) + (iz + 1); // checked!

	int lid3 = lid4 + ny * (nz + 1); // checked!
	int rid3 = rid4 + ny * (nz + 1); // checked!

	UExz[lid0] = UExz[lid0] * RBExz[lid1] + RAExz[lid2] * (Hz[lid3] - Hz[lid4]) / dx;
	UExz[rid0] = UExz[rid0] * RBExz[rid1] + RAExz[rid2] * (Hz[rid3] - Hz[rid4]) / dx;
}
dim3 gridUEzx(nx - 1, ny);
dim3 blockUEzx(npml - 1);
__global__ void gpu_UEzx(float *UEzx, float *RBEzx, float *RAEzx, float *Hx)
{
	/*
	dim3 blockUEzx(npml - 1);
	dim3 gridUEzx(nx - 1, ny);

	in0 UEzx  nx + 1     ny     nz + 1
	in1 RBEzx nx - 1     ny     2*(npml-1)
	in2 RAEzx nx - 1     ny     2*(npml-1)
	in3 Hx    nx + 1     ny     nz

	运算块大小 nx-1 * ny * npml-1

	UEzx(2:nx, :, [2:npml nz-npml+2:nz])=RBEzx .* UEzx(2:nx, :, [2:npml nz-npml+2:nz])...
	+RAEzx ./ dz .* (Hx(2:nx, :, [2:npml nz-npml+2:nz]) - Hx(2:nx, :, [1:npml-1 nz-npml+1:nz-1]));
	*/
	int ix = blockIdx.x;   // ix in [0, nx)
	int iy = blockIdx.y;   // iy in [0, npml - 1)
	int iz = threadIdx.x;  // iz in [0, nz)

	int lid0 = (ix + 1) * ny * (nz + 1) + iy * (nz + 1) + (iz + 1); // checked!
	int rid0 = (ix + 1) * ny * (nz + 1) + iy * (nz + 1) + (iz + 1 + nz - npml);  //checked!

	int lid1 = ix * ny * (2 * (npml - 1)) + iy * (2 * (npml - 1)) + iz; // checked!
	int rid1 = ix * ny * (2 * (npml - 1)) + iy * (2 * (npml - 1)) + (iz + npml - 1); // checked!

	int lid2 = lid1; // checked!
	int rid2 = rid1; // checked!

	int lid4 = (ix + 1) * ny * nz + iy * nz + iz; // checked!
	int rid4 = (ix + 1) * ny * nz + iy * nz + (iz + nz - npml); // checked!

	int lid3 = lid4 + 1; // checked!
	int rid3 = rid4 + 1; // checked!

	UEzx[lid0] = UEzx[lid0] * RBEzx[lid1] + RAEzx[lid2] * (Hx[lid3] - Hx[lid4]) / dz;
	UEzx[rid0] = UEzx[rid0] * RBEzx[rid1] + RAEzx[rid2] * (Hx[rid3] - Hx[rid4]) / dz;
}
dim3 gridUEzy(nx, ny - 1);
dim3 blockUEzy(npml - 1);
__global__ void gpu_UEzy(float *UEzy, float *RBEzy, float *RAEzy, float *Hy)
{
	/*
	dim3 blockUEzy(npml - 1);
	dim3 gridUEzy(nx, ny - 1);

	in0 UEzy  nx      ny + 1    nz + 1
	in1 RBEzy nx      ny - 1    2*(npml-1)
	in2 RAEzy nx      ny - 1    2*(npml-1)
	in3 Hy    nx      ny + 1    nz

	运算块大小 nx * ny - 1 * npml-1

	UEzy(:, 2:ny, [2:npml nz-npml+2:nz])=RBEzy.*UEzy(:, 2:ny, [2:npml nz-npml+2:nz])...
	+RAEzy./dz.*(Hy(:, 2:ny, [2:npml nz-npml+2:nz])-Hy(:, 2:ny, [1:npml-1 nz-npml+1:nz-1]));
	*/
	int ix = blockIdx.x;   // ix in [0, nx)
	int iy = blockIdx.y;   // iy in [0, npml - 1)
	int iz = threadIdx.x;  // iz in [0, nz)

	int lid0 = ix * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + (iz + 1); // checked!
	int rid0 = ix * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + (iz + 1 + nz - npml);  //checked!

	int lid1 = ix * (ny - 1) * (2 * (npml - 1)) + iy * (2 * (npml - 1)) + iz; // checked!
	int rid1 = ix * (ny - 1) * (2 * (npml - 1)) + iy * (2 * (npml - 1)) + (iz + npml - 1); // checked!

	int lid2 = lid1; // checked!
	int rid2 = rid1; // checked!

	int lid4 = ix * (ny + 1) * nz + (iy + 1) * nz + iz; // checked!
	int rid4 = ix * (ny + 1) * nz + (iy + 1) * nz + (iz + nz - npml); // checked!

	int lid3 = lid4 + 1; // checked!
	int rid3 = rid4 + 1; // checked!

	UEzy[lid0] = UEzy[lid0] * RBEzy[lid1] + RAEzy[lid2] * (Hy[lid3] - Hy[lid4]) / dz;
	UEzy[rid0] = UEzy[rid0] * RBEzy[rid1] + RAEzy[rid2] * (Hy[rid3] - Hy[rid4]) / dz;
}
dim3 gridEx(nx, ny - 1);
dim3 blockEx(nz - 1);
__global__ void gpu_Ex(float *Ex, float *CAEx, float *CBEx, float *ky_Ex, float *kz_Ex, float *Hz, float *Hy, float *UEyz, float *UEzy)
{
	//
	// * dim3 blockEx(nz-1);
	// * dim3 gridEx(nx, ny-1);
	// * 运算块大小 nx * ny-1 * nz-1
	// * Ex(:, 2:ny, 2:nz)
	//
	int ix = blockIdx.x;      // ix in [0, nx)
	int iy = blockIdx.y + 1;  // iy in [1, ny)
	int iz = threadIdx.x + 1; // iz in [1, nz)

	int idx = ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz;
	int idxHz = ix * ny * (nz + 1) + iy * (nz + 1) + iz;
	int idxHy = ix * (ny + 1)*nz + iy * nz + iz;
	int deltaHz = nz + 1;
	int deltaHy = 1;
	float CBE = CBEx[idx];

	Ex[idx] = Ex[idx] * CAEx[idx]
		+ CBE / ky_Ex[idx] * (Hz[idxHz] - Hz[idxHz - deltaHz]) / dy
		- CBE / kz_Ex[idx] * (Hy[idxHy] - Hy[idxHy - deltaHy]) / dz
		+ CBE * UEyz[idx]
		- CBE * UEzy[idx];
}
dim3 gridEy(nx - 1, ny);
dim3 blockEy(nz - 1);
__global__ void gpu_Ey(float *Ey, float *CAEy, float *CBEy, float *kz_Ey, float *kx_Ey, float *Hx, float *Hz, float *UEzx, float *UExz)
{
	//
	// * dim3 blockEy(nz-1);
	// * dim3 gridEy(nx-1, ny);
	// * 运算块大小 nx-1 * ny * nz-1
	// * Ey(2:nx, :, 2:nz)
	//
	int ix = blockIdx.x + 1;  // ix in [1, nx)
	int iy = blockIdx.y;      // iy in [0, ny)
	int iz = threadIdx.x + 1; // iz in [1, nz)

	int idx = ix * ny * (nz + 1) + iy * (nz + 1) + iz;
	int idxHx = ix * ny * nz + iy * nz + iz;
	int idxHz = ix * ny * (nz + 1) + iy * (nz + 1) + iz;
	int deltaHx = 1;
	int deltaHz = ny * (nz + 1);
	float CBE = CBEy[idx];

	Ey[idx] = Ey[idx] * CAEy[idx]
		+ CBE / kz_Ey[idx] * (Hx[idxHx] - Hx[idxHx - deltaHx]) / dz
		- CBE / kx_Ey[idx] * (Hz[idxHz] - Hz[idxHz - deltaHz]) / dx
		+ CBE * UEzx[idx]
		- CBE * UExz[idx];
}
dim3 gridEz(nx - 1, ny - 1);
dim3 blockEz(nz);
__global__ void gpu_Ez(float *Ez, float *CAEz, float *CBEz, float *kx_Ez, float *ky_Ez, float *Hy, float *Hx, float *UExy, float *UEyx)
{
	//
	// * dim3 blockEz(nz);
	// * dim3 gridEz(nx-1, ny-1);
	// * 运算块大小 nx-1 * ny-1 * nz
	// * Ez(2:nx, 2:ny, :)
	// * Ez大小为nx ny nz+1
	//
	int ix = blockIdx.x + 1; // ix in [1, nx)
	int iy = blockIdx.y + 1; // iy in [1, ny)
	int iz = threadIdx.x;    // iz in [0, nz)

	int idx = ix * (ny + 1) * nz + iy * nz + iz;
	int idxHy = ix * (ny + 1) * nz + iy * nz + iz;
	int idxHx = ix * ny * nz + iy * nz + iz;
	int deltaHy = (ny + 1) * nz;
	int deltaHx = nz;
	float CBE = CBEz[idx];

	Ez[idx] = Ez[idx] * CAEz[idx]
		+ CBE / kx_Ez[idx] * (Hy[idxHy] - Hy[idxHy - deltaHy]) / dx
		- CBE / ky_Ez[idx] * (Hx[idxHx] - Hx[idxHx - deltaHx]) / dy
		+ CBE * UExy[idx]
		- CBE * UEyx[idx];
}

dim3 grid_zheng_1(npmlc, ny - 2 * npml);
dim3 grid_zheng_2(nx - 2 * npml, npmlc);
dim3 grid_zheng_3(nx - 2 * npml, ny - 2 * npml);
dim3 grid_zheng_last(nx - 2 * npml, ny - 2 * npml);
dim3 block_zheng_1(nz - 2 * npml);
dim3 block_zheng_2(nz - 2 * npml);
dim3 block_zheng_3(npmlc);
dim3 block_zheng_last(nz - 2 * npml);
__global__ void gpu_zheng_1(
	float *dev_Ex_zheng, float *dev_Ey_zheng, float *dev_Ez_zheng,
	float *dev_Hx_zheng, float *dev_Hy_zheng, float *dev_Hz_zheng,
	float *dev_Ex, float *dev_Ey, float *dev_Ez,
	float *dev_Hx, float *dev_Hy, float *dev_Hz,
	int j)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = threadIdx.x;

	int lidzheng; //**_zheng_* 前半部分的位置
	int ridzheng; //**_zheng_* 后半部分的位置
	int lidEx, lidEy, lidEz, lidHx, lidHy, lidHz;
	int ridEx, ridEy, ridEz, ridHx, ridHy, ridHz;

	lidzheng =
		j * (2 * npmlc) * (ny - 2 * npml) * (nz - 2 * npml) +
		ix * (ny - 2 * npml) * (nz - 2 * npml) +
		iy * (nz - 2 * npml) +
		iz;
	lidEx =
		(ix + npml) * (ny + 1) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEy =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEz =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHx =
		(ix + npml) * (ny + 0) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHy =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHz =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);

	ridzheng = lidzheng + (ny - 2 * npml) * (nz - 2 * npml) * (npmlc);
	ridEx = lidEx + (ny + 1) * (nz + 1) * (nx - 2 * npml - npmlc);
	ridEy = lidEy + (ny + 0) * (nz + 1) * (nx - 2 * npml - npmlc);
	ridEz = lidEz + (ny + 1) * (nz + 0) * (nx - 2 * npml - npmlc);
	ridHx = lidHx + (ny + 0) * (nz + 0) * (nx - 2 * npml - npmlc);
	ridHy = lidHy + (ny + 1) * (nz + 0) * (nx - 2 * npml - npmlc);
	ridHz = lidHz + (ny + 0) * (nz + 1) * (nx - 2 * npml - npmlc);

	dev_Ex_zheng[lidzheng] = dev_Ex[lidEx];
	dev_Ey_zheng[lidzheng] = dev_Ey[lidEy];
	dev_Ez_zheng[lidzheng] = dev_Ez[lidEz];
	dev_Hx_zheng[lidzheng] = dev_Hx[lidHx];
	dev_Hy_zheng[lidzheng] = dev_Hy[lidHy];
	dev_Hz_zheng[lidzheng] = dev_Hz[lidHz];
	dev_Ex_zheng[ridzheng] = dev_Ex[ridEx];
	dev_Ey_zheng[ridzheng] = dev_Ey[ridEy];
	dev_Ez_zheng[ridzheng] = dev_Ez[ridEz];
	dev_Hx_zheng[ridzheng] = dev_Hx[ridHx];
	dev_Hy_zheng[ridzheng] = dev_Hy[ridHy];
	dev_Hz_zheng[ridzheng] = dev_Hz[ridHz];
}

__global__ void gpu_zheng_2(
	float *dev_Ex_zheng, float *dev_Ey_zheng, float *dev_Ez_zheng,
	float *dev_Hx_zheng, float *dev_Hy_zheng, float *dev_Hz_zheng,
	float *dev_Ex, float *dev_Ey, float *dev_Ez,
	float *dev_Hx, float *dev_Hy, float *dev_Hz,
	int j)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = threadIdx.x;

	int lidzheng; //**_zheng_* 前半部分的位置
	int ridzheng; //**_zheng_* 后半部分的位置
	int lidEx, lidEy, lidEz, lidHx, lidHy, lidHz;
	int ridEx, ridEy, ridEz, ridHx, ridHy, ridHz;

	lidzheng =
		j * (nx - 2 * npml) * (2 * npmlc) * (nz - 2 * npml) +
		ix * (2 * npmlc) * (nz - 2 * npml) +
		iy * (nz - 2 * npml) +
		iz;
	lidEx =
		(ix + npml) * (ny + 1) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEy =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEz =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHx =
		(ix + npml) * (ny + 0) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHy =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHz =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);


	ridzheng = lidzheng + (nz - 2 * npml) * (npmlc);
	ridEx = lidEx + (nz + 1) * (ny - 2 * npml - npmlc);
	ridEy = lidEy + (nz + 1) * (ny - 2 * npml - npmlc);
	ridEz = lidEz + (nz + 0) * (ny - 2 * npml - npmlc);
	ridHx = lidHx + (nz + 0) * (ny - 2 * npml - npmlc);
	ridHy = lidHy + (nz + 0) * (ny - 2 * npml - npmlc);
	ridHz = lidHz + (nz + 1) * (ny - 2 * npml - npmlc);

	dev_Ex_zheng[lidzheng] = dev_Ex[lidEx];
	dev_Ey_zheng[lidzheng] = dev_Ey[lidEy];
	dev_Ez_zheng[lidzheng] = dev_Ez[lidEz];
	dev_Hx_zheng[lidzheng] = dev_Hx[lidHx];
	dev_Hy_zheng[lidzheng] = dev_Hy[lidHy];
	dev_Hz_zheng[lidzheng] = dev_Hz[lidHz];
	dev_Ex_zheng[ridzheng] = dev_Ex[ridEx];
	dev_Ey_zheng[ridzheng] = dev_Ey[ridEy];
	dev_Ez_zheng[ridzheng] = dev_Ez[ridEz];
	dev_Hx_zheng[ridzheng] = dev_Hx[ridHx];
	dev_Hy_zheng[ridzheng] = dev_Hy[ridHy];
	dev_Hz_zheng[ridzheng] = dev_Hz[ridHz];
}

__global__ void gpu_zheng_3(
	float *dev_Ex_zheng, float *dev_Ey_zheng, float *dev_Ez_zheng,
	float *dev_Hx_zheng, float *dev_Hy_zheng, float *dev_Hz_zheng,
	float *dev_Ex, float *dev_Ey, float *dev_Ez,
	float *dev_Hx, float *dev_Hy, float *dev_Hz,
	int j)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = threadIdx.x;

	int lidzheng; //**_zheng_* 前半部分的位置
	int ridzheng; //**_zheng_* 后半部分的位置
	int lidEx, lidEy, lidEz, lidHx, lidHy, lidHz;
	int ridEx, ridEy, ridEz, ridHx, ridHy, ridHz;

	lidzheng =
		j * (nx - 2 * npml) * (ny - 2 * npml) * (2 * npmlc) +
		ix * (ny - 2 * npml) * (2 * npmlc) +
		iy * (2 * npmlc) +
		iz;
	lidEx =
		(ix + npml) * (ny + 1) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEy =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEz =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHx =
		(ix + npml) * (ny + 0) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHy =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHz =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);


	ridzheng = lidzheng + (npmlc);
	ridEx = lidEx + (nz - 2 * npml - npmlc);
	ridEy = lidEy + (nz - 2 * npml - npmlc);
	ridEz = lidEz + (nz - 2 * npml - npmlc);
	ridHx = lidHx + (nz - 2 * npml - npmlc);
	ridHy = lidHy + (nz - 2 * npml - npmlc);
	ridHz = lidHz + (nz - 2 * npml - npmlc);

	dev_Ex_zheng[lidzheng] = dev_Ex[lidEx];
	dev_Ey_zheng[lidzheng] = dev_Ey[lidEy];
	dev_Ez_zheng[lidzheng] = dev_Ez[lidEz];
	dev_Hx_zheng[lidzheng] = dev_Hx[lidHx];
	dev_Hy_zheng[lidzheng] = dev_Hy[lidHy];
	dev_Hz_zheng[lidzheng] = dev_Hz[lidHz];
	dev_Ex_zheng[ridzheng] = dev_Ex[ridEx];
	dev_Ey_zheng[ridzheng] = dev_Ey[ridEy];
	dev_Ez_zheng[ridzheng] = dev_Ez[ridEz];
	dev_Hx_zheng[ridzheng] = dev_Hx[ridHx];
	dev_Hy_zheng[ridzheng] = dev_Hy[ridHy];
	dev_Hz_zheng[ridzheng] = dev_Hz[ridHz];
}

__global__ void gpu_zheng_last(
	float *dev_Ex_zheng, float *dev_Ey_zheng, float *dev_Ez_zheng,
	float *dev_Hx_zheng, float *dev_Hy_zheng, float *dev_Hz_zheng,
	float *dev_Ex, float *dev_Ey, float *dev_Ez,
	float *dev_Hx, float *dev_Hy, float *dev_Hz)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = threadIdx.x;

	int lidzheng; //**_zheng_* 前半部分的位置
	int lidEx, lidEy, lidEz, lidHx, lidHy, lidHz;

	lidzheng =
		ix * (ny - 2 * npml) * (nz - 2 * npml) +
		iy * (nz - 2 * npml) +
		iz;
	lidEx =
		(ix + npml) * (ny + 1) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEy =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEz =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHx =
		(ix + npml) * (ny + 0) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHy =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHz =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);

	dev_Ex_zheng[lidzheng] = dev_Ex[lidEx];
	dev_Ey_zheng[lidzheng] = dev_Ey[lidEy];
	dev_Ez_zheng[lidzheng] = dev_Ez[lidEz];
	dev_Hx_zheng[lidzheng] = dev_Hx[lidHx];
	dev_Hy_zheng[lidzheng] = dev_Hy[lidHy];
	dev_Hz_zheng[lidzheng] = dev_Hz[lidHz];
}


__global__ void gpu_back_zheng_1(
	float *dev_Ex_zheng, float *dev_Ey_zheng, float *dev_Ez_zheng,
	float *dev_Hx_zheng, float *dev_Hy_zheng, float *dev_Hz_zheng,
	float *dev_Ex, float *dev_Ey, float *dev_Ez,
	float *dev_Hx, float *dev_Hy, float *dev_Hz,
	int j)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = threadIdx.x;

	int lidzheng; //**_zheng_* 前半部分的位置
	int ridzheng; //**_zheng_* 后半部分的位置
	int lidEx, lidEy, lidEz, lidHx, lidHy, lidHz;
	int ridEx, ridEy, ridEz, ridHx, ridHy, ridHz;

	lidzheng =
		j * (2 * npmlc) * (ny - 2 * npml) * (nz - 2 * npml) +
		ix * (ny - 2 * npml) * (nz - 2 * npml) +
		iy * (nz - 2 * npml) +
		iz;
	lidEx =
		(ix + npml) * (ny + 1) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEy =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEz =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHx =
		(ix + npml) * (ny + 0) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHy =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHz =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);

	ridzheng = lidzheng + (ny - 2 * npml) * (nz - 2 * npml) * (npmlc);
	ridEx = lidEx + (ny + 1) * (nz + 1) * (nx - 2 * npml - npmlc);
	ridEy = lidEy + (ny + 0) * (nz + 1) * (nx - 2 * npml - npmlc);
	ridEz = lidEz + (ny + 1) * (nz + 0) * (nx - 2 * npml - npmlc);
	ridHx = lidHx + (ny + 0) * (nz + 0) * (nx - 2 * npml - npmlc);
	ridHy = lidHy + (ny + 1) * (nz + 0) * (nx - 2 * npml - npmlc);
	ridHz = lidHz + (ny + 0) * (nz + 1) * (nx - 2 * npml - npmlc);

	dev_Ex[lidEx] = dev_Ex_zheng[lidzheng];
	dev_Ey[lidEy] = dev_Ey_zheng[lidzheng];
	dev_Ez[lidEz] = dev_Ez_zheng[lidzheng];
	dev_Hx[lidHx] = dev_Hx_zheng[lidzheng];
	dev_Hy[lidHy] = dev_Hy_zheng[lidzheng];
	dev_Hz[lidHz] = dev_Hz_zheng[lidzheng];
	dev_Ex[ridEx] = dev_Ex_zheng[ridzheng];
	dev_Ey[ridEy] = dev_Ey_zheng[ridzheng];
	dev_Ez[ridEz] = dev_Ez_zheng[ridzheng];
	dev_Hx[ridHx] = dev_Hx_zheng[ridzheng];
	dev_Hy[ridHy] = dev_Hy_zheng[ridzheng];
	dev_Hz[ridHz] = dev_Hz_zheng[ridzheng];
}

__global__ void gpu_back_zheng_2(
	float *dev_Ex_zheng, float *dev_Ey_zheng, float *dev_Ez_zheng,
	float *dev_Hx_zheng, float *dev_Hy_zheng, float *dev_Hz_zheng,
	float *dev_Ex, float *dev_Ey, float *dev_Ez,
	float *dev_Hx, float *dev_Hy, float *dev_Hz,
	int j)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = threadIdx.x;

	int lidzheng; //**_zheng_* 前半部分的位置
	int ridzheng; //**_zheng_* 后半部分的位置
	int lidEx, lidEy, lidEz, lidHx, lidHy, lidHz;
	int ridEx, ridEy, ridEz, ridHx, ridHy, ridHz;

	lidzheng =
		j * (nx - 2 * npml) * (2 * npmlc) * (nz - 2 * npml) +
		ix * (2 * npmlc) * (nz - 2 * npml) +
		iy * (nz - 2 * npml) +
		iz;
	lidEx =
		(ix + npml) * (ny + 1) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEy =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEz =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHx =
		(ix + npml) * (ny + 0) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHy =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHz =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);


	ridzheng = lidzheng + (nz - 2 * npml) * (npmlc);
	ridEx = lidEx + (nz + 1) * (ny - 2 * npml - npmlc);
	ridEy = lidEy + (nz + 1) * (ny - 2 * npml - npmlc);
	ridEz = lidEz + (nz + 0) * (ny - 2 * npml - npmlc);
	ridHx = lidHx + (nz + 0) * (ny - 2 * npml - npmlc);
	ridHy = lidHy + (nz + 0) * (ny - 2 * npml - npmlc);
	ridHz = lidHz + (nz + 1) * (ny - 2 * npml - npmlc);

	dev_Ex[lidEx] = dev_Ex_zheng[lidzheng];
	dev_Ey[lidEy] = dev_Ey_zheng[lidzheng];
	dev_Ez[lidEz] = dev_Ez_zheng[lidzheng];
	dev_Hx[lidHx] = dev_Hx_zheng[lidzheng];
	dev_Hy[lidHy] = dev_Hy_zheng[lidzheng];
	dev_Hz[lidHz] = dev_Hz_zheng[lidzheng];
	dev_Ex[ridEx] = dev_Ex_zheng[ridzheng];
	dev_Ey[ridEy] = dev_Ey_zheng[ridzheng];
	dev_Ez[ridEz] = dev_Ez_zheng[ridzheng];
	dev_Hx[ridHx] = dev_Hx_zheng[ridzheng];
	dev_Hy[ridHy] = dev_Hy_zheng[ridzheng];
	dev_Hz[ridHz] = dev_Hz_zheng[ridzheng];
}

__global__ void gpu_back_zheng_3(
	float *dev_Ex_zheng, float *dev_Ey_zheng, float *dev_Ez_zheng,
	float *dev_Hx_zheng, float *dev_Hy_zheng, float *dev_Hz_zheng,
	float *dev_Ex, float *dev_Ey, float *dev_Ez,
	float *dev_Hx, float *dev_Hy, float *dev_Hz,
	int j)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = threadIdx.x;

	int lidzheng; //**_zheng_* 前半部分的位置
	int ridzheng; //**_zheng_* 后半部分的位置
	int lidEx, lidEy, lidEz, lidHx, lidHy, lidHz;
	int ridEx, ridEy, ridEz, ridHx, ridHy, ridHz;

	lidzheng =
		j * (nx - 2 * npml) * (ny - 2 * npml) * (2 * npmlc) +
		ix * (ny - 2 * npml) * (2 * npmlc) +
		iy * (2 * npmlc) +
		iz;
	lidEx =
		(ix + npml) * (ny + 1) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEy =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEz =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHx =
		(ix + npml) * (ny + 0) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHy =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHz =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);


	ridzheng = lidzheng + (npmlc);
	ridEx = lidEx + (nz - 2 * npml - npmlc);
	ridEy = lidEy + (nz - 2 * npml - npmlc);
	ridEz = lidEz + (nz - 2 * npml - npmlc);
	ridHx = lidHx + (nz - 2 * npml - npmlc);
	ridHy = lidHy + (nz - 2 * npml - npmlc);
	ridHz = lidHz + (nz - 2 * npml - npmlc);

	dev_Ex[lidEx] = dev_Ex_zheng[lidzheng];
	dev_Ey[lidEy] = dev_Ey_zheng[lidzheng];
	dev_Ez[lidEz] = dev_Ez_zheng[lidzheng];
	dev_Hx[lidHx] = dev_Hx_zheng[lidzheng];
	dev_Hy[lidHy] = dev_Hy_zheng[lidzheng];
	dev_Hz[lidHz] = dev_Hz_zheng[lidzheng];
	dev_Ex[ridEx] = dev_Ex_zheng[ridzheng];
	dev_Ey[ridEy] = dev_Ey_zheng[ridzheng];
	dev_Ez[ridEz] = dev_Ez_zheng[ridzheng];
	dev_Hx[ridHx] = dev_Hx_zheng[ridzheng];
	dev_Hy[ridHy] = dev_Hy_zheng[ridzheng];
	dev_Hz[ridHz] = dev_Hz_zheng[ridzheng];
}

__global__ void gpu_back_zheng_last(
	float *dev_Ex_zheng, float *dev_Ey_zheng, float *dev_Ez_zheng,
	float *dev_Hx_zheng, float *dev_Hy_zheng, float *dev_Hz_zheng,
	float *dev_Ex, float *dev_Ey, float *dev_Ez,
	float *dev_Hx, float *dev_Hy, float *dev_Hz)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = threadIdx.x;

	int lidzheng; //**_zheng_* 前半部分的位置
	int lidEx, lidEy, lidEz, lidHx, lidHy, lidHz;

	lidzheng =
		ix * (ny - 2 * npml) * (nz - 2 * npml) +
		iy * (nz - 2 * npml) +
		iz;
	lidEx =
		(ix + npml) * (ny + 1) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEy =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	lidEz =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHx =
		(ix + npml) * (ny + 0) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHy =
		(ix + npml) * (ny + 1) * (nz + 0) +
		(iy + npml) * (nz + 0) +
		(iz + npml);
	lidHz =
		(ix + npml) * (ny + 0) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);

	dev_Ex[lidEx] = dev_Ex_zheng[lidzheng];
	dev_Ey[lidEy] = dev_Ey_zheng[lidzheng];
	dev_Ez[lidEz] = dev_Ez_zheng[lidzheng];
	dev_Hx[lidHx] = dev_Hx_zheng[lidzheng];
	dev_Hy[lidHy] = dev_Hy_zheng[lidzheng];
	dev_Hz[lidHz] = dev_Hz_zheng[lidzheng];
}

dim3 grid_fan_huanyuan(nx - 2 * npml, ny - 2 * npml);
dim3 block_fan_huanyuan(nz - 2 * npml);
__global__ void gpu_fan_huanyuan(float *dev_dst, float *dev_Ex)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = threadIdx.x;

	int lidfan, lidEx; //**_zheng_* 前半部分的位置

	lidfan =
		ix * (ny - 2 * npml) * (nz - 2 * npml) +
		iy * (nz - 2 * npml) +
		iz;
	lidEx =
		(ix + npml) * (ny + 1) * (nz + 1) +
		(iy + npml) * (nz + 1) +
		(iz + npml);
	dev_dst[lidfan] = dev_Ex[lidEx];
}


dim3 grid_HE1(nx - np - np, ny - np - np);
dim3 block_HE1(nz - np - np);
__global__ void gpu_H1(
	float *dev_Hx1, float *dev_Hy1, float *dev_Hz1, 
	float *dev_Ex1, float *dev_Ey1, float *dev_Ez1, 
	float *dev_CPHx, float *dev_CPHy, float *dev_CPHz,
	float *dev_CQHx, float *dev_CQHy, float *dev_CQHz)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = blockIdx.z;
	int idxHx1 = (ix + np)*(ny + 0)*(nz + 0) + (iy + np)*(nz + 0) + (iz + np);
	int idxHy1 = (ix + np)*(ny + 1)*(nz + 0) + (iy + np)*(nz + 0) + (iz + np);
	int idxHz1 = (ix + np)*(ny + 0)*(nz + 1) + (iy + np)*(nz + 1) + (iz + np);
	int idxEx1 = (ix + np)*(ny + 1)*(nz + 1) + (iy + np)*(nz + 1) + (iz + np);
	int idxEy1 = (ix + np)*(ny + 0)*(nz + 1) + (iy + np)*(nz + 1) + (iz + np);
	int idxEz1 = (ix + np)*(ny + 1)*(nz + 0) + (iy + np)*(nz + 0) + (iz + np);
	int delEz1_Hx1 = nz;
	int delEy1_Hx1 = 1;
	int delEx1_Hy1 = 1;
	int delEz1_Hy1 = (ny + 1) * nz;
	int delEy1_Hz1 = ny * (nz + 1);
	int delEx1_Hz1 = nz + 1;

	const float rfCPHx = 1 / dev_CPHx[idxHx1];// 倒数reciprocal of fCPHx
	const float fCQHx = dev_CQHx[idxHx1]; 
	dev_Hx1[idxHx1] = rfCPHx * dev_Hx1[idxHx1]
		+ rfCPHx * fCQHx / dy * (dev_Ez1[idxEz1 + delEz1_Hx1] - dev_Ez1[idxEz1])
		- rfCPHx * fCQHx / dz * (dev_Ey1[idxEy1 + delEy1_Hx1] - dev_Ey1[idxEy1]);

	const float rfCPHy = 1 / dev_CPHy[idxHy1];// 倒数reciprocal of fCPHy
	const float fCQHy = dev_CQHy[idxHy1];
	dev_Hy1[idxHy1] = rfCPHy * dev_Hy1[idxHy1]
		+ rfCPHy * fCQHy / dz * (dev_Ex1[idxEx1 + delEx1_Hy1] - dev_Ex1[idxEx1])
		- rfCPHy * fCQHy / dx * (dev_Ez1[idxEz1 + delEz1_Hy1] - dev_Ez1[idxEz1]);

	const float rfCPHz = 1 / dev_CPHz[idxHz1];// 倒数reciprocal of fCPHz
	const float fCQHz = dev_CQHz[idxHz1];
	dev_Hz1[idxHz1] = rfCPHz * dev_Hz1[idxHz1]
		+ rfCPHz * fCQHz / dx * (dev_Ey1[idxEy1 + delEy1_Hz1] - dev_Ey1[idxEy1])
		- rfCPHz * fCQHz / dy * (dev_Ex1[idxEx1 + delEx1_Hz1] - dev_Ex1[idxEx1]);

}

__global__ void gpu_E1(
	float *dev_Hx1, float *dev_Hy1, float *dev_Hz1,
	float *dev_Ex1, float *dev_Ey1, float *dev_Ez1,
	float *dev_CAEx, float *dev_CAEy, float *dev_CAEz,
	float *dev_CBEx, float *dev_CBEy, float *dev_CBEz)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y;
	int iz = blockIdx.z;
	int idxHx1 = (ix + np)*(ny + 0)*(nz + 0) + (iy + np)*(nz + 0) + (iz + np);
	int idxHy1 = (ix + np)*(ny + 1)*(nz + 0) + (iy + np)*(nz + 0) + (iz + np);
	int idxHz1 = (ix + np)*(ny + 0)*(nz + 1) + (iy + np)*(nz + 1) + (iz + np);
	int idxEx1 = (ix + np)*(ny + 1)*(nz + 1) + (iy + np)*(nz + 1) + (iz + np);
	int idxEy1 = (ix + np)*(ny + 0)*(nz + 1) + (iy + np)*(nz + 1) + (iz + np);
	int idxEz1 = (ix + np)*(ny + 1)*(nz + 0) + (iy + np)*(nz + 0) + (iz + np);
	int delHz1_Ex1 = nz + 1;
	int delHy1_Ex1 = 1;
	int delHx1_Ey1 = 1;
	int delHz1_Ey1 = ny * (nz + 1);
	int delHy1_Ez1 = (ny + 1) * nz;
	int delHx1_Ez1 = nz;

	const float rfCAEx = 1 / dev_CAEx[idxEx1];// 倒数reciprocal of fCAEx
	const float fCBEx = dev_CBEx[idxEx1];
	dev_Ex1[idxEx1] = rfCAEx * dev_Ex1[idxEx1]
		+ rfCAEx * fCBEx / dy * (dev_Hz1[idxHz1] - dev_Hz1[idxHz1 - delHz1_Ex1])
		- rfCAEx * fCBEx / dz * (dev_Hy1[idxHy1] - dev_Hy1[idxHy1 - delHy1_Ex1]);

	const float rfCAEy = 1 / dev_CAEy[idxEy1];// 倒数reciprocal of fCAEy
	const float fCBEy = dev_CBEy[idxEy1];
	dev_Ey1[idxEy1] = rfCAEy * dev_Ey1[idxEy1]
		+ rfCAEy * fCBEy / dz * (dev_Hx1[idxHx1] - dev_Hx1[idxHx1 - delHx1_Ey1])
		- rfCAEy * fCBEy / dx * (dev_Hz1[idxHz1] - dev_Hz1[idxHz1 - delHz1_Ey1]);

	const float rfCAEz = 1 / dev_CAEz[idxEz1];// 倒数reciprocal of fCAEz
	const float fCBEz = dev_CBEz[idxEz1];
	dev_Ez1[idxEz1] = rfCAEz * dev_Ez1[idxEz1]
		+ rfCAEz * fCBEz / dx * (dev_Hy1[idxHy1] - dev_Hy1[idxHy1 - delHy1_Ez1])
		- rfCAEz * fCBEz / dy * (dev_Hx1[idxHx1] - dev_Hx1[idxHx1 - delHx1_Ez1]);

}

dim3 grid_nzf(nx - 2 * npml, ny - 2 * npml);
dim3 block_nzf(nz - 2 * npml);
__global__ void gpu_nzf(float *dev_dst, float *dev_src1, float *dev_src2)
{
	int idx =
		blockIdx.x * (ny - 2 * npml) * (nz - 2 * npml) +
		blockIdx.y * (nz - 2 * npml) +
		threadIdx.x;
	dev_dst[idx] += dev_src1[idx] * dev_src2[idx];
}

void read_int(const char *name, int *a, int n1, int n2, int n3)
{
	FILE *fp = fopen(name, "r");
	if (fp == NULL) // 判断文件读入是否正确
	{
		printf("fopen %s error! \n", name);
		return;
	}
	printf("fopen %s ok! \n", name);
	for (int i = 0; i < n1; i++)
	{
		for (int k = 0; k < n3; k++)
		{
			for (int j = 0; j < n2; j++)
			{
				fscanf(fp, "%d", &a[i * n2*n3 + j * n3 + k]); // 读入a[i][j][k]

			}
		}
	}
	printf("read %s OK\n", name);

	fclose(fp);
	return;
}

void read_float(const char *name, float *a, int n1, int n2, int n3)
{
	FILE *fp = fopen(name, "r");
	if (fp == NULL) // 判断文件读入是否正确
	{
		printf("fopen %s error! \n", name);
		return;
	}
	printf("fopen %s ok! \n", name);
	for (int i = 0; i < n1; i++)
	{
		for (int k = 0; k < n3; k++)
		{
			for (int j = 0; j < n2; j++)
			{
				fscanf(fp, "%f", a + i * n2*n3 + j * n3 + k); // 读入a[i][j][k]			
			}

		}
	}
	printf("read %s OK\n", name);

	fclose(fp);
	return;
}

void print_nzf(const char *name, float *a, int n1, int n2, int n3)
{
	FILE *fp = fopen(name, "w+");
	if (fp == NULL) // 判断文件读入是否正确
	{
		printf("fopen %s error! \n", name);
		return;
	}
	printf("fopen %s ok! \n", name);
	for (int k = 0; k < n3; k++)
	{
		for (int j = 0; j < n2; j++)
		{
			for (int i = 0; i < n1; i++)
			{
				fprintf(fp, "%8f ", *(a + i * n2*n3 + j * n3 + k)); // 输出a[i][j][k]			
			}

		}
	}
	printf("print %s OK\n", name);

	fclose(fp);
	return;
}

void read_data_from_txt()
{
	if (isPianYi)
	{
		read_float("data_pianyi/CAEx.txt", (float*)CAEx, nx, ny + 1, nz + 1);
		read_float("data_pianyi/CBEx.txt", (float*)CBEx, nx, ny + 1, nz + 1);
		read_float("data_pianyi/RAEyz.txt", (float*)RAEyz, nx, 2 * (npml - 1), nz - 1);
		read_float("data_pianyi/RBEyz.txt", (float*)RBEyz, nx, 2 * (npml - 1), nz - 1);
		read_float("data_pianyi/RAEzy.txt", (float*)RAEzy, nx, ny - 1, 2 * (npml - 1));
		read_float("data_pianyi/RBEzy.txt", (float*)RBEzy, nx, ny - 1, 2 * (npml - 1));
		read_float("data_pianyi/CAEy.txt", (float*)CAEy, nx + 1, ny, nz + 1);
		read_float("data_pianyi/CBEy.txt", (float*)CBEy, nx + 1, ny, nz + 1);
		read_float("data_pianyi/RAEzx.txt", (float*)RAEzx, nx - 1, ny, 2 * (npml - 1));
		read_float("data_pianyi/RBEzx.txt", (float*)RBEzx, nx - 1, ny, 2 * (npml - 1));
		read_float("data_pianyi/RAExz.txt", (float*)RAExz, 2 * (npml - 1), ny, nz - 1);
		read_float("data_pianyi/RBExz.txt", (float*)RBExz, 2 * (npml - 1), ny, nz - 1);
		read_float("data_pianyi/CAEz.txt", (float*)CAEz, nx + 1, ny + 1, nz);
		read_float("data_pianyi/CBEz.txt", (float*)CBEz, nx + 1, ny + 1, nz);
		read_float("data_pianyi/RAExy.txt", (float*)RAExy, 2 * (npml - 1), ny - 1, nz);
		read_float("data_pianyi/RBExy.txt", (float*)RBExy, 2 * (npml - 1), ny - 1, nz);
		read_float("data_pianyi/RAEyx.txt", (float*)RAEyx, nx - 1, 2 * (npml - 1), nz);
		read_float("data_pianyi/RBEyx.txt", (float*)RBEyx, nx - 1, 2 * (npml - 1), nz);

		read_float("data_pianyi/CPHx.txt", (float*)CPHx, nx + 1, ny, nz);
		read_float("data_pianyi/CQHx.txt", (float*)CQHx, nx + 1, ny, nz);
		read_float("data_pianyi/RAHyz.txt", (float*)RAHyz, nx - 1, 2 * npml, nz);
		read_float("data_pianyi/RBHyz.txt", (float*)RBHyz, nx - 1, 2 * npml, nz);
		read_float("data_pianyi/RAHzy.txt", (float*)RAHzy, nx - 1, ny, 2 * npml);
		read_float("data_pianyi/RBHzy.txt", (float*)RBHzy, nx - 1, ny, 2 * npml);
		read_float("data_pianyi/CPHy.txt", (float*)CPHy, nx, ny + 1, nz);
		read_float("data_pianyi/CQHy.txt", (float*)CQHy, nx, ny + 1, nz);
		read_float("data_pianyi/RAHzx.txt", (float*)RAHzx, nx, ny - 1, 2 * npml);
		read_float("data_pianyi/RBHzx.txt", (float*)RBHzx, nx, ny - 1, 2 * npml);
		read_float("data_pianyi/RAHxz.txt", (float*)RAHxz, 2 * npml, ny - 1, nz);
		read_float("data_pianyi/RBHxz.txt", (float*)RBHxz, 2 * npml, ny - 1, nz);
		read_float("data_pianyi/CPHz.txt", (float*)CPHz, nx, ny, nz + 1);
		read_float("data_pianyi/CQHz.txt", (float*)CQHz, nx, ny, nz + 1);
		read_float("data_pianyi/RAHxy.txt", (float*)RAHxy, 2 * npml, ny, nz - 1);
		read_float("data_pianyi/RBHxy.txt", (float*)RBHxy, 2 * npml, ny, nz - 1);
		read_float("data_pianyi/RAHyx.txt", (float*)RAHyx, nx, 2 * npml, nz - 1);
		read_float("data_pianyi/RBHyx.txt", (float*)RBHyx, nx, 2 * npml, nz - 1);

		read_float("data_pianyi/kx_Ey.txt", (float*)kx_Ey, nx + 1, ny, nz + 1);
		read_float("data_pianyi/kx_Ez.txt", (float*)kx_Ez, nx + 1, ny + 1, nz);
		read_float("data_pianyi/ky_Ex.txt", (float*)ky_Ex, nx, ny + 1, nz + 1);
		read_float("data_pianyi/ky_Ez.txt", (float*)ky_Ez, nx + 1, ny + 1, nz);
		read_float("data_pianyi/kz_Ex.txt", (float*)kz_Ex, nx, ny + 1, nz + 1);
		read_float("data_pianyi/kz_Ey.txt", (float*)kz_Ey, nx + 1, ny, nz + 1);

		read_float("data_pianyi/kx_Hy.txt", (float*)kx_Hy, nx, ny + 1, nz);
		read_float("data_pianyi/kx_Hz.txt", (float*)kx_Hz, nx, ny, nz + 1);
		read_float("data_pianyi/ky_Hx.txt", (float*)ky_Hx, nx + 1, ny, nz);
		read_float("data_pianyi/ky_Hz.txt", (float*)ky_Hz, nx, ny, nz + 1);
		read_float("data_pianyi/kz_Hx.txt", (float*)kz_Hx, nx + 1, ny, nz);
		read_float("data_pianyi/kz_Hy.txt", (float*)kz_Hy, nx, ny + 1, nz);

		read_int("data_pianyi/fswzx.txt", (int*)fswzx, 1, 1, szfsw);
		read_int("data_pianyi/fswzy.txt", (int*)fswzy, 1, 1, szfsw);
		read_int("data_pianyi/fswzz.txt", (int*)fswzz, 1, 1, szfsw);
		read_int("data_pianyi/jswzx.txt", (int*)jswzx, 1, 1, szfsw);
		read_int("data_pianyi/jswzy.txt", (int*)jswzy, 1, 1, szfsw);
		read_int("data_pianyi/jswzz.txt", (int*)jswzz, 1, 1, szfsw);
		read_float("data_pianyi/source.txt", (float*)source, 1, 1, it);
		read_float("data_pianyi/E_obs.txt", (float*)source, 1, it, szfsw);
	}
	else
	{
		read_float("data_zhengyan/CAEx.txt", (float*)CAEx, nx, ny + 1, nz + 1);
		read_float("data_zhengyan/CBEx.txt", (float*)CBEx, nx, ny + 1, nz + 1);
		read_float("data_zhengyan/RAEyz.txt", (float*)RAEyz, nx, 2 * (npml - 1), nz - 1);
		read_float("data_zhengyan/RBEyz.txt", (float*)RBEyz, nx, 2 * (npml - 1), nz - 1);
		read_float("data_zhengyan/RAEzy.txt", (float*)RAEzy, nx, ny - 1, 2 * (npml - 1));
		read_float("data_zhengyan/RBEzy.txt", (float*)RBEzy, nx, ny - 1, 2 * (npml - 1));
		read_float("data_zhengyan/CAEy.txt", (float*)CAEy, nx + 1, ny, nz + 1);
		read_float("data_zhengyan/CBEy.txt", (float*)CBEy, nx + 1, ny, nz + 1);
		read_float("data_zhengyan/RAEzx.txt", (float*)RAEzx, nx - 1, ny, 2 * (npml - 1));
		read_float("data_zhengyan/RBEzx.txt", (float*)RBEzx, nx - 1, ny, 2 * (npml - 1));
		read_float("data_zhengyan/RAExz.txt", (float*)RAExz, 2 * (npml - 1), ny, nz - 1);
		read_float("data_zhengyan/RBExz.txt", (float*)RBExz, 2 * (npml - 1), ny, nz - 1);
		read_float("data_zhengyan/CAEz.txt", (float*)CAEz, nx + 1, ny + 1, nz);
		read_float("data_zhengyan/CBEz.txt", (float*)CBEz, nx + 1, ny + 1, nz);
		read_float("data_zhengyan/RAExy.txt", (float*)RAExy, 2 * (npml - 1), ny - 1, nz);
		read_float("data_zhengyan/RBExy.txt", (float*)RBExy, 2 * (npml - 1), ny - 1, nz);
		read_float("data_zhengyan/RAEyx.txt", (float*)RAEyx, nx - 1, 2 * (npml - 1), nz);
		read_float("data_zhengyan/RBEyx.txt", (float*)RBEyx, nx - 1, 2 * (npml - 1), nz);

		read_float("data_zhengyan/CPHx.txt", (float*)CPHx, nx + 1, ny, nz);
		read_float("data_zhengyan/CQHx.txt", (float*)CQHx, nx + 1, ny, nz);
		read_float("data_zhengyan/RAHyz.txt", (float*)RAHyz, nx - 1, 2 * npml, nz);
		read_float("data_zhengyan/RBHyz.txt", (float*)RBHyz, nx - 1, 2 * npml, nz);
		read_float("data_zhengyan/RAHzy.txt", (float*)RAHzy, nx - 1, ny, 2 * npml);
		read_float("data_zhengyan/RBHzy.txt", (float*)RBHzy, nx - 1, ny, 2 * npml);
		read_float("data_zhengyan/CPHy.txt", (float*)CPHy, nx, ny + 1, nz);
		read_float("data_zhengyan/CQHy.txt", (float*)CQHy, nx, ny + 1, nz);
		read_float("data_zhengyan/RAHzx.txt", (float*)RAHzx, nx, ny - 1, 2 * npml);
		read_float("data_zhengyan/RBHzx.txt", (float*)RBHzx, nx, ny - 1, 2 * npml);
		read_float("data_zhengyan/RAHxz.txt", (float*)RAHxz, 2 * npml, ny - 1, nz);
		read_float("data_zhengyan/RBHxz.txt", (float*)RBHxz, 2 * npml, ny - 1, nz);
		read_float("data_zhengyan/CPHz.txt", (float*)CPHz, nx, ny, nz + 1);
		read_float("data_zhengyan/CQHz.txt", (float*)CQHz, nx, ny, nz + 1);
		read_float("data_zhengyan/RAHxy.txt", (float*)RAHxy, 2 * npml, ny, nz - 1);
		read_float("data_zhengyan/RBHxy.txt", (float*)RBHxy, 2 * npml, ny, nz - 1);
		read_float("data_zhengyan/RAHyx.txt", (float*)RAHyx, nx, 2 * npml, nz - 1);
		read_float("data_zhengyan/RBHyx.txt", (float*)RBHyx, nx, 2 * npml, nz - 1);

		read_float("data_zhengyan/kx_Ey.txt", (float*)kx_Ey, nx + 1, ny, nz + 1);
		read_float("data_zhengyan/kx_Ez.txt", (float*)kx_Ez, nx + 1, ny + 1, nz);
		read_float("data_zhengyan/ky_Ex.txt", (float*)ky_Ex, nx, ny + 1, nz + 1);
		read_float("data_zhengyan/ky_Ez.txt", (float*)ky_Ez, nx + 1, ny + 1, nz);
		read_float("data_zhengyan/kz_Ex.txt", (float*)kz_Ex, nx, ny + 1, nz + 1);
		read_float("data_zhengyan/kz_Ey.txt", (float*)kz_Ey, nx + 1, ny, nz + 1);

		read_float("data_zhengyan/kx_Hy.txt", (float*)kx_Hy, nx, ny + 1, nz);
		read_float("data_zhengyan/kx_Hz.txt", (float*)kx_Hz, nx, ny, nz + 1);
		read_float("data_zhengyan/ky_Hx.txt", (float*)ky_Hx, nx + 1, ny, nz);
		read_float("data_zhengyan/ky_Hz.txt", (float*)ky_Hz, nx, ny, nz + 1);
		read_float("data_zhengyan/kz_Hx.txt", (float*)kz_Hx, nx + 1, ny, nz);
		read_float("data_zhengyan/kz_Hy.txt", (float*)kz_Hy, nx, ny + 1, nz);

		read_int("data_zhengyan/fswzx.txt", (int*)fswzx, 1, 1, szfsw);
		read_int("data_zhengyan/fswzy.txt", (int*)fswzy, 1, 1, szfsw);
		read_int("data_zhengyan/fswzz.txt", (int*)fswzz, 1, 1, szfsw);
		read_int("data_zhengyan/jswzx.txt", (int*)jswzx, 1, 1, szfsw);
		read_int("data_zhengyan/jswzy.txt", (int*)jswzy, 1, 1, szfsw);
		read_int("data_zhengyan/jswzz.txt", (int*)jswzz, 1, 1, szfsw);
		read_float("data_zhengyan/source.txt", (float*)source, 1, 1, it);
	}
}

void print_E_obs()
{
	const char *name = "output/E_obs.txt";
	FILE *fp = fopen(name, "w+");
	if (fp == NULL) // 判断文件读入是否正确
	{
		printf("fopen %s error! \n", name);
	}
	printf("print fopen %s ok! \n", name);

	fprintf(fp, "输出E_obs[%d][%d]\n", it, szfsw);
	fprintf(fp, "共有 %d 行 %d 列 \n", szfsw, it);

	for (int i = 0; i < szfsw; i++)
	{
		for (int j = 0; j < it; j++)
		{
			fprintf(fp, "%8f ", E_obs[j][i]);
		}
		fprintf(fp, "\n");
	}
	printf("print %s OK\n", name);

	fclose(fp);
	return;
}

void gpu_memory_malloc()
{
	cudaError_t cudaStatus = cudaSuccess;
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");  }
	//原来内存中存在的数组，数组大小用内存数组大小就行
	cudaStatus = cudaMalloc((void**)&dev_CAEx, sizeof(CAEx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_CBEx, sizeof(CBEx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAEyz, sizeof(RAEyz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBEyz, sizeof(RBEyz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAEzy, sizeof(RAEzy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBEzy, sizeof(RBEzy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_CAEy, sizeof(CAEy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_CBEy, sizeof(CBEy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAExz, sizeof(RAExz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBExz, sizeof(RBExz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAEzx, sizeof(RAEzx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBEzx, sizeof(RBEzx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_CAEz, sizeof(CAEz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_CBEz, sizeof(CBEz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAExy, sizeof(RAExy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBExy, sizeof(RBExy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAEyx, sizeof(RAEyx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBEyx, sizeof(RBEyx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_CPHx, sizeof(CPHx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_CQHx, sizeof(CQHx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAHyz, sizeof(RAHyz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBHyz, sizeof(RBHyz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAHzy, sizeof(RAHzy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBHzy, sizeof(RBHzy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_CPHy, sizeof(CPHy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_CQHy, sizeof(CQHy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAHxz, sizeof(RAHxz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBHxz, sizeof(RBHxz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAHzx, sizeof(RAHzx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBHzx, sizeof(RBHzx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_CPHz, sizeof(CPHz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_CQHz, sizeof(CQHz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAHxy, sizeof(RAHxy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBHxy, sizeof(RBHxy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RAHyx, sizeof(RAHyx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_RBHyx, sizeof(RBHyx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_kx_Ey, sizeof(kx_Ey));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_kx_Ez, sizeof(kx_Ez));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_ky_Ex, sizeof(ky_Ex));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_ky_Ez, sizeof(ky_Ez));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_kz_Ex, sizeof(kz_Ex));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_kz_Ey, sizeof(kz_Ey));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_kx_Hy, sizeof(kx_Hy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_kx_Hz, sizeof(kx_Hz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_ky_Hx, sizeof(ky_Hx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_ky_Hz, sizeof(ky_Hz));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_kz_Hx, sizeof(kz_Hx));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_kz_Hy, sizeof(kz_Hy));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	//gpu显存新创建数组，原来内存中不存在
	int szEx = nx * (ny + 1)*(nz + 1);
	int szEy = (nx + 1)*ny*(nz + 1);
	int szEz = (nx + 1)*(ny + 1)*nz;
	int szHx = (nx + 1)*ny*nz;
	int szHy = nx * (ny + 1)*nz;
	int szHz = nx * ny*(nz + 1);

	cudaStatus = cudaMalloc((void**)&dev_Ex, szEx * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UEyz, szEx * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UEzy, szEx * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Ey, szEy * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UEzx, szEy * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UExz, szEy * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Ez, szEz * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UExy, szEz * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UEyx, szEz * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Hx, szHx * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UHyz, szHx * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UHzy, szHx * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Hy, szHy * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UHzx, szHy * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UHxz, szHy * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Hz, szHz * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UHxy, szHz * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_UHyx, szHz * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_V, sizeof(V));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_E_obs, sizeof(E_obs));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_source, sizeof(source));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	// 第二部分并行需要用到的变量

	cudaStatus = cudaMalloc((void**)&dev_fan, sizeof(fan));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_huanyuan, sizeof(huanyuan));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_ns, sizeof(ns));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_zv, sizeof(zv));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_fv, sizeof(fv));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Ex1, sizeof(Ex1));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Ey1, sizeof(Ey1));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Ez1, sizeof(Ez1));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Hx1, sizeof(Hx1));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Hy1, sizeof(Hy1));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Hz1, sizeof(Hz1));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc failed!");}

	// 超大数组

	cudaStatus = cudaMalloc((void**)&dev_Ex_zheng_1, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Ex_zheng_2, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Ex_zheng_3, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Ey_zheng_1, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Ey_zheng_2, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Ey_zheng_3, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Ez_zheng_1, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Ez_zheng_2, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Ez_zheng_3, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Hx_zheng_1, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Hx_zheng_2, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Hx_zheng_3, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Hy_zheng_1, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Hy_zheng_2, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Hy_zheng_3, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Hz_zheng_1, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Hz_zheng_2, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Hz_zheng_3, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Ex_zheng_last, (nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Ey_zheng_last, (nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Ez_zheng_last, (nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}

	cudaStatus = cudaMalloc((void**)&dev_Hx_zheng_last, (nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Hy_zheng_last, (nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
	cudaStatus = cudaMalloc((void**)&dev_Hz_zheng_last, (nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc Super Big Array failed!");}
}

// flag == 0 将GPU显存中的E*, UE**, H*, UH**, (V, E_obs)置零
// flag == 1 将GPU显存中的E*, UE**, H*, UH**, (V, E*_zheng_*, H*_zheng_*, E*_zheng_last, H*_zheng_last, fan, huanyuan)置零
// flag == 2 将GPU显存中的E*, UE**, H*, UH**, (V, E*1, H*1)置零
void gpu_memory_set_zero(int flag)
{
	int szEx = nx * (ny + 1)*(nz + 1);
	int szEy = (nx + 1)*ny*(nz + 1);
	int szEz = (nx + 1)*(ny + 1)*nz;
	int szHx = (nx + 1)*ny*nz;
	int szHy = nx * (ny + 1)*nz;
	int szHz = nx * ny*(nz + 1);

	cudaMemset(dev_Ex, 0, szEx * sizeof(float));
	cudaMemset(dev_UEyz, 0, szEx * sizeof(float));
	cudaMemset(dev_UEzy, 0, szEx * sizeof(float));

	cudaMemset(dev_Ey, 0, szEy * sizeof(float));
	cudaMemset(dev_UEzx, 0, szEy * sizeof(float));
	cudaMemset(dev_UExz, 0, szEy * sizeof(float));

	cudaMemset(dev_Ez, 0, szEz * sizeof(float));
	cudaMemset(dev_UExy, 0, szEz * sizeof(float));
	cudaMemset(dev_UEyx, 0, szEz * sizeof(float));

	cudaMemset(dev_Hx, 0, szHx * sizeof(float));
	cudaMemset(dev_UHyz, 0, szHx * sizeof(float));
	cudaMemset(dev_UHzy, 0, szHx * sizeof(float));

	cudaMemset(dev_Hy, 0, szHy * sizeof(float));
	cudaMemset(dev_UHzx, 0, szHy * sizeof(float));
	cudaMemset(dev_UHxz, 0, szHy * sizeof(float));

	cudaMemset(dev_Hz, 0, szHz * sizeof(float));
	cudaMemset(dev_UHxy, 0, szHz * sizeof(float));
	cudaMemset(dev_UHyx, 0, szHz * sizeof(float));

	if (flag == 0)
	{
		cudaMemset(dev_V, 0, sizeof(V));
		cudaMemset(dev_E_obs, 0, sizeof(E_obs));
	}
	else if (flag == 1)
	{
		cudaMemset(dev_V, 0, sizeof(V));

		cudaMemset(dev_Ex_zheng_1, 0, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Ex_zheng_2, 0, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Ex_zheng_3, 0, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

		cudaMemset(dev_Ey_zheng_1, 0, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Ey_zheng_2, 0, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Ey_zheng_3, 0, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

		cudaMemset(dev_Ez_zheng_1, 0, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Ez_zheng_2, 0, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Ez_zheng_3, 0, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

		cudaMemset(dev_Hx_zheng_1, 0, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Hx_zheng_2, 0, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Hx_zheng_3, 0, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

		cudaMemset(dev_Hy_zheng_1, 0, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Hy_zheng_2, 0, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Hy_zheng_3, 0, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

		cudaMemset(dev_Hz_zheng_1, 0, (it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Hz_zheng_2, 0, (it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
		cudaMemset(dev_Hz_zheng_3, 0, (it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

		size_t sz_last = (nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float);
		cudaMemset(dev_Ex_zheng_last, 0, sz_last);
		cudaMemset(dev_Ey_zheng_last, 0, sz_last);
		cudaMemset(dev_Ez_zheng_last, 0, sz_last);

		cudaMemset(dev_Hx_zheng_last, 0, sz_last);
		cudaMemset(dev_Hy_zheng_last, 0, sz_last);
		cudaMemset(dev_Hz_zheng_last, 0, sz_last);

		cudaMemset(dev_fan, 0, sizeof(fan));
		cudaMemset(dev_huanyuan, 0, sizeof(huanyuan));
	}
	else
	{
		cudaMemset(dev_Ex1, 0, sizeof(Ex1));
		cudaMemset(dev_Ey1, 0, sizeof(Ey1));
		cudaMemset(dev_Ez1, 0, sizeof(Ez1));

		cudaMemset(dev_Hx1, 0, sizeof(Hx1));
		cudaMemset(dev_Hy1, 0, sizeof(Hy1));
		cudaMemset(dev_Hz1, 0, sizeof(Hz1));
	}
}

// 将内存中的变量复制到显存中
// flag == 0 CAE CBE RAE RBE CPH CQH RAH RBH k*_E* k*_H* source
// flag == 1 CAE CBE RAE RBE CPH CQH RAH RBH k*_E* k*_H* source
void gpu_memory_copy()
{
	cudaError_t cudaStatus;
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_CAEx, CAEx, sizeof(CAEx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_CBEx, CBEx, sizeof(CBEx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAEyz, RAEyz, sizeof(RAEyz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBEyz, RBEyz, sizeof(RBEyz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAEzy, RAEzy, sizeof(RAEzy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBEzy, RBEzy, sizeof(RBEzy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}

	cudaStatus = cudaMemcpy(dev_CAEy, CAEy, sizeof(CAEy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_CBEy, CBEy, sizeof(CBEy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAExz, RAExz, sizeof(RAExz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBExz, RBExz, sizeof(RBExz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAEzx, RAEzx, sizeof(RAEzx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBEzx, RBEzx, sizeof(RBEzx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}

	cudaStatus = cudaMemcpy(dev_CAEz, CAEz, sizeof(CAEz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_CBEz, CBEz, sizeof(CBEz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAExy, RAExy, sizeof(RAExy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBExy, RBExy, sizeof(RBExy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAEyx, RAEyx, sizeof(RAEyx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBEyx, RBEyx, sizeof(RBEyx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}

	cudaStatus = cudaMemcpy(dev_CPHx, CPHx, sizeof(CPHx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_CQHx, CQHx, sizeof(CQHx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAHyz, RAHyz, sizeof(RAHyz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBHyz, RBHyz, sizeof(RBHyz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAHzy, RAHzy, sizeof(RAHzy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBHzy, RBHzy, sizeof(RBHzy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}

	cudaStatus = cudaMemcpy(dev_CPHy, CPHy, sizeof(CPHy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_CQHy, CQHy, sizeof(CQHy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAHxz, RAHxz, sizeof(RAHxz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBHxz, RBHxz, sizeof(RBHxz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAHzx, RAHzx, sizeof(RAHzx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBHzx, RBHzx, sizeof(RBHzx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}

	cudaStatus = cudaMemcpy(dev_CPHz, CPHz, sizeof(CPHz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_CQHz, CQHz, sizeof(CQHz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAHxy, RAHxy, sizeof(RAHxy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBHxy, RBHxy, sizeof(RBHxy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RAHyx, RAHyx, sizeof(RAHyx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_RBHyx, RBHyx, sizeof(RBHyx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}

	cudaStatus = cudaMemcpy(dev_kx_Ey, kx_Ey, sizeof(kx_Ey), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_kx_Ez, kx_Ez, sizeof(kx_Ez), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_ky_Ex, ky_Ex, sizeof(ky_Ex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_ky_Ez, ky_Ez, sizeof(ky_Ez), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_kz_Ex, kz_Ex, sizeof(kz_Ex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_kz_Ey, kz_Ey, sizeof(kz_Ey), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}

	cudaStatus = cudaMemcpy(dev_kx_Hy, kx_Hy, sizeof(kx_Hy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_kx_Hz, kx_Hz, sizeof(kx_Hz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_ky_Hx, ky_Hx, sizeof(ky_Hx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_ky_Hz, ky_Hz, sizeof(ky_Hz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_kz_Hx, kz_Hx, sizeof(kz_Hx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	cudaStatus = cudaMemcpy(dev_kz_Hy, kz_Hy, sizeof(kz_Hy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}

	cudaStatus = cudaMemcpy(dev_source, source, sizeof(source), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!");}
	if (isPianYi)
	{
		cudaStatus = cudaMemcpy(dev_E_obs, E_obs, sizeof(E_obs), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); }
	}
}

// 释放显存空间
void gpu_memory_free()
{
	cudaFree(dev_Ex);
	cudaFree(dev_Ey);
	cudaFree(dev_Ez);

	cudaFree(dev_UEyz);
	cudaFree(dev_UEzy);
	cudaFree(dev_UExz);
	cudaFree(dev_UEzx);
	cudaFree(dev_UExy);
	cudaFree(dev_UEyx);

	cudaFree(dev_Hx);
	cudaFree(dev_Hy);
	cudaFree(dev_Hz);

	cudaFree(dev_UHyz);
	cudaFree(dev_UHzy);
	cudaFree(dev_UHxz);
	cudaFree(dev_UHzx);
	cudaFree(dev_UHxy);
	cudaFree(dev_UHyx);

	cudaFree(dev_CAEx);
	cudaFree(dev_CAEy);
	cudaFree(dev_CAEz);

	cudaFree(dev_CBEx);
	cudaFree(dev_CBEy);
	cudaFree(dev_CBEz);

	cudaFree(dev_RAEyz);
	cudaFree(dev_RAEzy);
	cudaFree(dev_RAEzx);
	cudaFree(dev_RAExz);
	cudaFree(dev_RAExy);
	cudaFree(dev_RAEyx);

	cudaFree(dev_RBEyz);
	cudaFree(dev_RBEzy);
	cudaFree(dev_RBEzx);
	cudaFree(dev_RBExz);
	cudaFree(dev_RBExy);
	cudaFree(dev_RBEyx);

	cudaFree(dev_CPHx);
	cudaFree(dev_CQHx);
	cudaFree(dev_CPHy);
	cudaFree(dev_CQHy);
	cudaFree(dev_CPHz);
	cudaFree(dev_CQHz);

	cudaFree(dev_RAHyz);
	cudaFree(dev_RAHzy);
	cudaFree(dev_RAHzx);
	cudaFree(dev_RAHxz);
	cudaFree(dev_RAHxy);
	cudaFree(dev_RAHyx);

	cudaFree(dev_RBHyz);
	cudaFree(dev_RBHzy);
	cudaFree(dev_RBHzx);
	cudaFree(dev_RBHxz);
	cudaFree(dev_RBHxy);
	cudaFree(dev_RBHyx);


	cudaFree(fswzx);
	cudaFree(fswzy);
	cudaFree(fswzz);
	cudaFree(jswzx);
	cudaFree(jswzy);
	cudaFree(jswzz);

	cudaFree(dev_E_obs);
	cudaFree(dev_V);
	cudaFree(dev_source);

	cudaFree(dev_kx_Ey);
	cudaFree(dev_kx_Ez);
	cudaFree(dev_ky_Ex);
	cudaFree(dev_ky_Ez);
	cudaFree(dev_kz_Ex);
	cudaFree(dev_kz_Ey);

	cudaFree(dev_kx_Hy);
	cudaFree(dev_kx_Hz);
	cudaFree(dev_ky_Hx);
	cudaFree(dev_ky_Hz);
	cudaFree(dev_kz_Hx);
	cudaFree(dev_kz_Hy);

	cudaFree(dev_Ex_zheng_1);
	cudaFree(dev_Ex_zheng_2);
	cudaFree(dev_Ex_zheng_3);

	cudaFree(dev_Ey_zheng_1);
	cudaFree(dev_Ey_zheng_2);
	cudaFree(dev_Ey_zheng_3);

	cudaFree(dev_Ez_zheng_1);
	cudaFree(dev_Ez_zheng_2);
	cudaFree(dev_Ez_zheng_3);

	cudaFree(dev_Hx_zheng_1);
	cudaFree(dev_Hx_zheng_2);
	cudaFree(dev_Hx_zheng_3);

	cudaFree(dev_Hy_zheng_1);
	cudaFree(dev_Hy_zheng_2);
	cudaFree(dev_Hy_zheng_3);

	cudaFree(dev_Hz_zheng_1);
	cudaFree(dev_Hz_zheng_2);
	cudaFree(dev_Hz_zheng_3);

	cudaFree(dev_Ex_zheng_last);
	cudaFree(dev_Ey_zheng_last);
	cudaFree(dev_Ez_zheng_last);

	cudaFree(dev_Hx_zheng_last);
	cudaFree(dev_Hy_zheng_last);
	cudaFree(dev_Hz_zheng_last);

	cudaFree(dev_fan);
	cudaFree(dev_huanyuan);
	cudaFree(dev_ns);
	cudaFree(dev_zv);
	cudaFree(dev_fv);
}

// gpu并行计算UH H UE E
void zheng_yan()
{
	cudaError_t cudaStatus = cudaSuccess;

	gpu_UHyz << < gridUHyz, blockUHyz >> > (dev_UHyz, dev_RBHyz, dev_RAHyz, dev_Ez);
	gpu_UHzy << < gridUHzy, blockUHzy >> > (dev_UHzy, dev_RBHzy, dev_RAHzy, dev_Ey);
	gpu_UHxy << < gridUHxy, blockUHxy >> > (dev_UHxy, dev_RBHxy, dev_RAHxy, dev_Ey);
	gpu_UHxz << < gridUHxz, blockUHxz >> > (dev_UHxz, dev_RBHxz, dev_RAHxz, dev_Ez);
	gpu_UHyx << < gridUHyx, blockUHyx >> > (dev_UHyx, dev_RBHyx, dev_RAHyx, dev_Ex);
	gpu_UHzx << < gridUHzx, blockUHzx >> > (dev_UHzx, dev_RBHzx, dev_RAHzx, dev_Ex);

	gpu_Hx << < gridHx, blockHx >> > (dev_Hx, dev_CPHx, dev_CQHx, dev_ky_Hx, dev_kz_Hx, dev_Ez, dev_Ey, dev_UHyz, dev_UHzy);
	gpu_Hy << < gridHy, blockHy >> > (dev_Hy, dev_CPHy, dev_CQHy, dev_kz_Hy, dev_kx_Hy, dev_Ex, dev_Ez, dev_UHzx, dev_UHxz);
	gpu_Hz << < gridHz, blockHz >> > (dev_Hz, dev_CPHz, dev_CQHz, dev_kx_Hz, dev_ky_Hz, dev_Ey, dev_Ex, dev_UHxy, dev_UHyx);

	gpu_UExy << < gridUExy, blockUExy >> > (dev_UExy, dev_RBExy, dev_RAExy, dev_Hy);
	gpu_UExz << < gridUExz, blockUExz >> > (dev_UExz, dev_RBExz, dev_RAExz, dev_Hz);
	gpu_UEyx << < gridUEyx, blockUEyx >> > (dev_UEyx, dev_RBEyx, dev_RAEyx, dev_Hx);
	gpu_UEyz << < gridUEyz, blockUEyz >> > (dev_UEyz, dev_RBEyz, dev_RAEyz, dev_Hz);
	gpu_UEzx << < gridUEzx, blockUEzx >> > (dev_UEzx, dev_RBEzx, dev_RAEzx, dev_Hx);
	gpu_UEzy << < gridUEzy, blockUEzy >> > (dev_UEzy, dev_RBEzy, dev_RAEzy, dev_Hy);

	gpu_Ex << < gridEx, blockEx >> > (dev_Ex, dev_CAEx, dev_CBEx, dev_ky_Ex, dev_kz_Ex, dev_Hz, dev_Hy, dev_UEyz, dev_UEzy);
	gpu_Ey << < gridEy, blockEy >> > (dev_Ey, dev_CAEy, dev_CBEy, dev_kz_Ey, dev_kx_Ey, dev_Hx, dev_Hz, dev_UEzx, dev_UExz);
	gpu_Ez << < gridEz, blockEz >> > (dev_Ez, dev_CAEz, dev_CBEz, dev_kx_Ez, dev_ky_Ez, dev_Hy, dev_Hx, dev_UExy, dev_UEyx);

	// 计算过程是否出错?
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("Zhengyan Calc Failed: %s\n", cudaGetErrorString(cudaStatus));
	}
}


cudaError_t gpu_parallel_one()
{
	cudaError_t cudaStatus = cudaSuccess;

	int i, j;
	for (i = 0; i < szfsw; i++)
	{
		gpu_memory_set_zero(0);	// flag == 0 将GPU显存中的E*, UE**, H*, UH**, (V, E_obs)置零

		for (j = 0; j < it; j++)
		{
			if (j % 10 == 0)
			{
				printf("i = %3d / %d,  j = %4d / %d\n", i, szfsw, j, it);
			}

			// matlab: Ex(fswzx(i),fswzy(i),fswzz(i))=source(j); 显存到显存
			int idxEx = (fswzx[i] - 1) * (ny + 1) * (nz + 1) + (fswzy[i] - 1) * (nz + 1) + (fswzz[i] - 1);
			cudaStatus = cudaMemcpy(&(dev_Ex[idxEx]), &(dev_source[j]), sizeof(float), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) { printf("source --> Ex cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus)); return cudaStatus; };

			// 调用GPU运算正演
			zheng_yan();

			// matlab: V(j)=Ex(jswzx(i), jswzy(i), jswzz(i)); 显存到显存
			idxEx = (jswzx[i] - 1) * (ny + 1) * (nz + 1) + (jswzy[i] - 1) * (nz + 1) + (jswzz[i] - 1);
			cudaStatus = cudaMemcpy(&(dev_V[j]), &(dev_Ex[idxEx]), sizeof(float), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) { printf("Ex --> V cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus)); return cudaStatus; };

			// matlab: E_obs(j,i) = V(j) 显存到内存
			cudaStatus = cudaMemcpy(&(E_obs[j][i]), &(dev_V[j]), sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) { printf("V --> E_obs cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus)); return cudaStatus; };
		}
	}

	cudaDeviceSynchronize();

	printf("finish calc 1 !\n");

	// 输出结果
	print_E_obs();

	return cudaStatus;
}

cudaError_t gpu_parallel_two()
{
	cudaError_t cudaStatus = cudaSuccess;
	cudaMemset(dev_ns, 0, sizeof(ns));
	cudaMemset(dev_zv, 0, sizeof(zv));
	cudaMemset(dev_fv, 0, sizeof(fv));
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("ns&zv&fv cudaMemset Failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	int i, j;
	for (i = 0; i < szfsw; i++)
	{
		// 111111
		gpu_memory_set_zero(1); // flag == 1 将GPU显存中的E*, UE**, H*, UH**, (V, E*_zheng_*, H*_zheng_*, E*_zheng_last, H*_zheng_last, fan, huanyuan)置零
		for (j = 0; j < it; j++)
		{
			if (j % 50 == 0) { printf("i = %3d / %d,  j = %4d / %d\n", i, szfsw, j, it); }

			// 调用GPU运算正演
			zheng_yan();

			gpu_zheng_1 << <grid_zheng_1, block_zheng_1 >> > (
				dev_Ex_zheng_1, dev_Ey_zheng_1, dev_Ez_zheng_1,
				dev_Hx_zheng_1, dev_Hy_zheng_1, dev_Hz_zheng_1,
				dev_Ex, dev_Ey, dev_Ez,
				dev_Hx, dev_Hy, dev_Hz,
				j);

			gpu_zheng_2 << <grid_zheng_2, block_zheng_2 >> > (
				dev_Ex_zheng_2, dev_Ey_zheng_2, dev_Ez_zheng_2,
				dev_Hx_zheng_2, dev_Hy_zheng_2, dev_Hz_zheng_2,
				dev_Ex, dev_Ey, dev_Ez,
				dev_Hx, dev_Hy, dev_Hz,
				j);

			gpu_zheng_3 << <grid_zheng_3, block_zheng_3 >> > (
				dev_Ex_zheng_3, dev_Ey_zheng_3, dev_Ez_zheng_3,
				dev_Hx_zheng_3, dev_Hy_zheng_3, dev_Hz_zheng_3,
				dev_Ex, dev_Ey, dev_Ez,
				dev_Hx, dev_Hy, dev_Hz,
				j);

			gpu_zheng_last << <grid_zheng_last, block_zheng_last >> > (
				dev_Ex_zheng_last, dev_Ey_zheng_last, dev_Ez_zheng_last,
				dev_Hx_zheng_last, dev_Hy_zheng_last, dev_Hz_zheng_last,
				dev_Ex, dev_Ey, dev_Ez,
				dev_Hx, dev_Hy, dev_Hz);
		}

		// 222222
		gpu_memory_set_zero(2);
		for (j = it - 1; j >= 0; j--)
		{
			//if (j % 50 == 0) { printf("i = %3d / %d,  j = %4d / %d\n", i, szfsw, j, it); }

			//Ex(fswzx(i), fswzy(i), fswzz(i)) = E_obs(j, i);
			int idxEx = (fswzx[i] - 1) * (ny + 1) * (nz + 1) + (fswzy[i] - 1) * (nz + 1) + (fswzz[i] - 1);
			int idxE_obs = j * szfsw + i;
			cudaStatus = cudaMemcpy(&(dev_Ex[idxEx]), &(dev_E_obs[idxE_obs]), sizeof(float), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) 
			{ 
				printf("E_obs --> Ex cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
				return cudaStatus; 
			}

			// 调用GPU运算正演
			zheng_yan();

			// matlab: fan=Ex(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
			gpu_fan_huanyuan << <grid_fan_huanyuan, block_fan_huanyuan >> > (dev_fan, dev_Ex);

			if (j == it - 1)
			{
				gpu_back_zheng_last << <grid_zheng_last, block_zheng_last >> > (
					dev_Ex_zheng_last, dev_Ey_zheng_last, dev_Ez_zheng_last,
					dev_Hx_zheng_last, dev_Hy_zheng_last, dev_Hz_zheng_last,
					dev_Ex1, dev_Ey1, dev_Ez1,
					dev_Hx1, dev_Hy1, dev_Hz1);
			}
			else //j < it - 1
			{
				gpu_back_zheng_1 << <grid_zheng_1, block_zheng_1 >> > (
					dev_Ex_zheng_1, dev_Ey_zheng_1, dev_Ez_zheng_1,
					dev_Hx_zheng_1, dev_Hy_zheng_1, dev_Hz_zheng_1,
					dev_Ex1, dev_Ey1, dev_Ez1,
					dev_Hx1, dev_Hy1, dev_Hz1,
					j);

				gpu_back_zheng_2 << <grid_zheng_2, block_zheng_2 >> > (
					dev_Ex_zheng_2, dev_Ey_zheng_2, dev_Ez_zheng_2,
					dev_Hx_zheng_2, dev_Hy_zheng_2, dev_Hz_zheng_2,
					dev_Ex1, dev_Ey1, dev_Ez1,
					dev_Hx1, dev_Hy1, dev_Hz1,
					j);

				gpu_back_zheng_3 << <grid_zheng_3, block_zheng_3 >> > (
					dev_Ex_zheng_3, dev_Ey_zheng_3, dev_Ez_zheng_3,
					dev_Hx_zheng_3, dev_Hy_zheng_3, dev_Hz_zheng_3,
					dev_Ex1, dev_Ey1, dev_Ez1,
					dev_Hx1, dev_Hy1, dev_Hz1,
					j);

				// matlab: Ex1(fswzx(i), fswzy(i), fswzz(i)) = source(j);
				int idxEx1 = (fswzx[i] - 1) * (ny + 1) * (nz + 1) + (fswzy[i] - 1) * (nz + 1) + (fswzz[i] - 1);
				cudaStatus = cudaMemcpy(&(dev_Ex1[idxEx1]), &(dev_source[j]), sizeof(float), cudaMemcpyDeviceToDevice);
				if (cudaStatus != cudaSuccess) 
				{ 
					printf("source --> Ex1 cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus)); 
					return cudaStatus; 
				}

				gpu_H1 << <grid_HE1, block_HE1 >> > (
					dev_Hx1, dev_Hy1, dev_Hz1,
					dev_Ex1, dev_Ey1, dev_Ez1,
					dev_CPHx, dev_CPHy, dev_CPHz,
					dev_CQHx, dev_CQHy, dev_CQHz);
				gpu_E1 << <grid_HE1, block_HE1 >> > (
					dev_Hx1, dev_Hy1, dev_Hz1,
					dev_Ex1, dev_Ey1, dev_Ez1,
					dev_CAEx, dev_CAEy, dev_CAEz,
					dev_CBEx, dev_CBEy, dev_CBEz);
			}

			// matlab: huanyuan=Ex1(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
			gpu_fan_huanyuan << <grid_fan_huanyuan, block_fan_huanyuan >> > (dev_huanyuan, dev_Ex);
			gpu_nzf << <grid_nzf, block_nzf >> > (dev_ns, dev_huanyuan, dev_fan);
			gpu_nzf << <grid_nzf, block_nzf >> > (dev_zv, dev_huanyuan, dev_huanyuan);
			gpu_nzf << <grid_nzf, block_nzf >> > (dev_fv, dev_fan, dev_fan);
		}
	}
	cudaStatus = cudaMemcpy(ns, dev_ns, sizeof(ns), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		printf("dev_ns --> ns cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaMemcpy(fv, dev_fv, sizeof(fv), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		printf("dev_fv --> fv cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaMemcpy(zv, dev_zv, sizeof(zv), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		printf("dev_zv --> ns cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}



	cudaDeviceSynchronize();

	printf("finish calc 2!\n");

	print_nzf("nzf/ns.txt", (float*)ns, nx - 2 * npml, ny - 2 * npml, nz - 2 * npml);
	print_nzf("nzf/fv.txt", (float*)fv, nx - 2 * npml, ny - 2 * npml, nz - 2 * npml);
	print_nzf("nzf/zv.txt", (float*)zv, nx - 2 * npml, ny - 2 * npml, nz - 2 * npml);

	return cudaStatus;
}

/************************************************************************************
* 主函数
************************************************************************************/
int main()
{
	// 切换工作目录
	chdir(path); //linux
	//_chdir(path);
	char str[80];
	printf("Current Dir: %s \n",getcwd(str, 80)); //linux
	//printf("Current Dir: %s \n", _getcwd(str, 80));
	if (Hz_zheng_3 == NULL)
	{
		printf("malloc failed! \n");
		return 1;
	}
	else
	{
		printf("addr of Hz_zheng_3 is %p\n", Hz_zheng_3);
	}
	// 从matlab输出的文本文件中读取数据
	read_data_from_txt();
	printf("Read Data From Txt OK ! \n");

	// 选择运算使用的GPU
	cudaError_t cudaStatus = cudaSetDevice(cudaDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); return 1; }
	else { printf("cudaSetDevice success!\n"); }

	// 分配显存，把数据从内存传输到显存
	gpu_memory_malloc();
	gpu_memory_copy();

	// 调用gpu运算并输出到文件
	if (isPianYi)
	{
		cudaStatus = gpu_parallel_two();
		if (cudaStatus != cudaSuccess) { printf("gpu_parallel_two failed!"); return 1; }
		else { printf("gpu_parallel_two success!\n"); }
	}
	else
	{
		cudaStatus = gpu_parallel_one();
		if (cudaStatus != cudaSuccess) { printf("gpu_parallel_one failed!"); return 1; }
		else { printf("gpu_parallel_one success!\n"); }
	}


	// 释放显存空间
	gpu_memory_free();

	// 重置GPU
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) { printf("cudaDeviceReset failed!"); return 1; }

	// 释放内存空间
	freeMemory();
	return 0;
}