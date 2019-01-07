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
#include "global_variables.cpp"


/************************************************************************************
* 函数定义
************************************************************************************/

dim3 blockUHyz(nz);
dim3 gridUHyz(npml, nx - 1); //npml: blockIdx.x的变化范围， nx-1就是: blockIdx.y的变化范围
__global__ void calcUHyz(float *UHyz, float *RBHyz, float *RAHyz, float *Ez, const float dy)
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

dim3 blockUHzy(npml);
dim3 gridUHzy(nx - 1, ny);
__global__ void calcUHzy(float *UHzy, float *RBHzy, float *RAHzy, float *Ey, const float dz)
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

dim3 blockUHzx(npml);
dim3 gridUHzx(nx, ny - 1);
__global__ void calcUHzx(float *UHzx, float *RBHzx, float *RAHzx, float *Ex, const float dz)
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

dim3 blockUHxz(nz);
dim3 gridUHxz(npml, ny - 1);
__global__ void calcUHxz(float *UHxz, float *RBHxz, float *RAHxz, float *Ez, const float dx)
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

dim3 blockUHxy(nz - 1);
dim3 gridUHxy(npml, ny);
__global__ void calcUHxy(float *UHxy, float *RBHxy, float *RAHxy, float *Ey, const float dx)
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

dim3 blockUHyx(nz - 1);
dim3 gridUHyx(npml, nx);
__global__ void calcUHyx(float *UHyx, float *RBHyx, float *RAHyx, float *Ex, const float dy)
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

dim3 blockHx(nz);
dim3 gridHx(nx - 1, ny);
__global__ void calcHx(float *Hx, float *CPHx, float *CQHx, float *ky_Hx, float *kz_Hx, float *Ez, float *Ey, float *UHyz, float *UHzy, const float dy, const float dz)
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

dim3 blockHy(nz);
dim3 gridHy(nx, ny - 1);
__global__ void calcHy(float *Hy, float *CPHy, float *CQHy, float *kz_Hy, float *kx_Hy, float *Ex, float *Ez, float *UHzx, float *UHxz, const float dz, const float dx)
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

dim3 blockHz(nz - 1);
dim3 gridHz(nx, ny);
__global__ void calcHz(float *Hz, float *CPHz, float *CQHz, float *kx_Hz, float *ky_Hz, float *Ey, float *Ex, float *UHxy, float *UHyx, const float dx, const float dy)
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

dim3 blockUEyz(nz - 1);
dim3 gridUEyz(npml - 1, nx);
__global__ void calcUEyz(float *UEyz, float *RBEyz, float *RAEyz, float *Hz, const float dy)
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

dim3 blockUEyx(nz - 1);
dim3 gridUEyx(npml - 1, nx);
__global__ void calcUEyx(float *UEyx, float *RBEyx, float *RAEyx, float *Hx, const float dy)
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

dim3 blockUExy(nz);
dim3 gridUExy(npml - 1, ny - 1);
__global__ void calcUExy(float *UExy, float *RBExy, float *RAExy, float *Hy, const float dx)
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

dim3 blockUExz(nz - 1);
dim3 gridUExz(npml - 1, ny);
__global__ void calcUExz(float *UExz, float *RBExz, float *RAExz, float *Hz, const float dx)
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

dim3 blockUEzx(npml - 1);
dim3 gridUEzx(nx - 1, ny);
__global__ void calcUEzx(float *UEzx, float *RBEzx, float *RAEzx, float *Hx, const float dz)
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

dim3 blockUEzy(npml - 1);
dim3 gridUEzy(nx, ny - 1);
__global__ void calcUEzy(float *UEzy, float *RBEzy, float *RAEzy, float *Hy, const float dz)
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

dim3 blockEx(nz - 1);
dim3 gridEx(nx, ny - 1);
__global__ void calcEx(float *Ex, float *CAEx, float *CBEx, float *ky_Ex, float *kz_Ex, float *Hz, float *Hy, float *UEyz, float *UEzy, const float dy, const float dz)
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

dim3 blockEy(nz - 1);
dim3 gridEy(nx - 1, ny);
__global__ void calcEy(float *Ey, float *CAEy, float *CBEy, float *kz_Ey, float *kx_Ey, float *Hx, float *Hz, float *UEzx, float *UExz, const float dz, const float dx)
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

dim3 blockEz(nz);
dim3 gridEz(nx - 1, ny - 1);
__global__ void calcEz(float *Ez, float *CAEz, float *CBEz, float *kx_Ez, float *ky_Ez, float *Hy, float *Hx, float *UExy, float *UEyx, const float dx, const float dy)
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

__global__ void print_dev_matrix(float *A, int i,int j,int k,int xdim,int ydim,int zdim)
{
	int	idx = i * ydim*zdim + j * zdim + k;
	printf("dev_Matrix[%d][%d][%d] = %8f\n", i, j, k, A[idx]);
}

void readInteger(const char *name, int *a, int n1, int n2, int n3)
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

void readFloat(const char *name, float *a, int n1, int n2, int n3)
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

void readAllData()
{

	readFloat("data/CAEx.txt", (float*)CAEx, nx, ny + 1, nz + 1);
	readFloat("data/CBEx.txt", (float*)CBEx, nx, ny + 1, nz + 1);
	readFloat("data/RAEyz.txt", (float*)RAEyz, nx, 2 * (npml - 1), nz - 1);
	readFloat("data/RBEyz.txt", (float*)RBEyz, nx, 2 * (npml - 1), nz - 1);
	readFloat("data/RAEzy.txt", (float*)RAEzy, nx, ny - 1, 2 * (npml - 1));
	readFloat("data/RBEzy.txt", (float*)RBEzy, nx, ny - 1, 2 * (npml - 1));
	readFloat("data/CAEy.txt", (float*)CAEy, nx + 1, ny, nz + 1);
	readFloat("data/CBEy.txt", (float*)CBEy, nx + 1, ny, nz + 1);
	readFloat("data/RAEzx.txt", (float*)RAEzx, nx - 1, ny, 2 * (npml - 1));
	readFloat("data/RBEzx.txt", (float*)RBEzx, nx - 1, ny, 2 * (npml - 1));
	readFloat("data/RAExz.txt", (float*)RAExz, 2 * (npml - 1), ny, nz - 1);
	readFloat("data/RBExz.txt", (float*)RBExz, 2 * (npml - 1), ny, nz - 1);
	readFloat("data/CAEz.txt", (float*)CAEz, nx + 1, ny + 1, nz);
	readFloat("data/CBEz.txt", (float*)CBEz, nx + 1, ny + 1, nz);
	readFloat("data/RAExy.txt", (float*)RAExy, 2 * (npml - 1), ny - 1, nz);
	readFloat("data/RBExy.txt", (float*)RBExy, 2 * (npml - 1), ny - 1, nz);
	readFloat("data/RAEyx.txt", (float*)RAEyx, nx - 1, 2 * (npml - 1), nz);
	readFloat("data/RBEyx.txt", (float*)RBEyx, nx - 1, 2 * (npml - 1), nz);

	readFloat("data/CPHx.txt", (float*)CPHx, nx + 1, ny, nz);
	readFloat("data/CQHx.txt", (float*)CQHx, nx + 1, ny, nz);
	readFloat("data/RAHyz.txt", (float*)RAHyz, nx - 1, 2 * npml, nz);
	readFloat("data/RBHyz.txt", (float*)RBHyz, nx - 1, 2 * npml, nz);
	readFloat("data/RAHzy.txt", (float*)RAHzy, nx - 1, ny, 2 * npml);
	readFloat("data/RBHzy.txt", (float*)RBHzy, nx - 1, ny, 2 * npml);
	readFloat("data/CPHy.txt", (float*)CPHy, nx, ny + 1, nz);
	readFloat("data/CQHy.txt", (float*)CQHy, nx, ny + 1, nz);
	readFloat("data/RAHzx.txt", (float*)RAHzx, nx, ny - 1, 2 * npml);
	readFloat("data/RBHzx.txt", (float*)RBHzx, nx, ny - 1, 2 * npml);
	readFloat("data/RAHxz.txt", (float*)RAHxz, 2 * npml, ny - 1, nz);
	readFloat("data/RBHxz.txt", (float*)RBHxz, 2 * npml, ny - 1, nz);
	readFloat("data/CPHz.txt", (float*)CPHz, nx, ny, nz + 1);
	readFloat("data/CQHz.txt", (float*)CQHz, nx, ny, nz + 1);
	readFloat("data/RAHxy.txt", (float*)RAHxy, 2 * npml, ny, nz - 1);
	readFloat("data/RBHxy.txt", (float*)RBHxy, 2 * npml, ny, nz - 1);
	readFloat("data/RAHyx.txt", (float*)RAHyx, nx, 2 * npml, nz - 1);
	readFloat("data/RBHyx.txt", (float*)RBHyx, nx, 2 * npml, nz - 1);

	readFloat("data/kx_Ey.txt", (float*)kx_Ey, nx + 1, ny, nz + 1);
	readFloat("data/kx_Ez.txt", (float*)kx_Ez, nx + 1, ny + 1, nz);
	readFloat("data/ky_Ex.txt", (float*)ky_Ex, nx, ny + 1, nz + 1);
	readFloat("data/ky_Ez.txt", (float*)ky_Ez, nx + 1, ny + 1, nz);
	readFloat("data/kz_Ex.txt", (float*)kz_Ex, nx, ny + 1, nz + 1);
	readFloat("data/kz_Ey.txt", (float*)kz_Ey, nx + 1, ny, nz + 1);

	readFloat("data/kx_Hy.txt", (float*)kx_Hy, nx, ny + 1, nz);
	readFloat("data/kx_Hz.txt", (float*)kx_Hz, nx, ny, nz + 1);
	readFloat("data/ky_Hx.txt", (float*)ky_Hx, nx + 1, ny, nz);
	readFloat("data/ky_Hz.txt", (float*)ky_Hz, nx, ny, nz + 1);
	readFloat("data/kz_Hx.txt", (float*)kz_Hx, nx + 1, ny, nz);
	readFloat("data/kz_Hy.txt", (float*)kz_Hy, nx, ny + 1, nz);

	readInteger("data/fswzx.txt", (int*)fswzx, 1, 1, szfsw);
	readInteger("data/fswzy.txt", (int*)fswzy, 1, 1, szfsw);
	readInteger("data/fswzz.txt", (int*)fswzz, 1, 1, szfsw);
	readInteger("data/jswzx.txt", (int*)jswzx, 1, 1, szfsw);
	readInteger("data/jswzy.txt", (int*)jswzy, 1, 1, szfsw);
	readInteger("data/jswzz.txt", (int*)jswzz, 1, 1, szfsw);
	readFloat("data/source.txt", (float*)source, 1, 1, it);
}

void printE_obs()
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
	//原来内存中存在的数组，数组大小用内存数组大小就行
	cudaMalloc((void**)&dev_CAEx, sizeof(CAEx));
	cudaMalloc((void**)&dev_CBEx, sizeof(CBEx));
	cudaMalloc((void**)&dev_RAEyz, sizeof(RAEyz));
	cudaMalloc((void**)&dev_RBEyz, sizeof(RBEyz));
	cudaMalloc((void**)&dev_RAEzy, sizeof(RAEzy));
	cudaMalloc((void**)&dev_RBEzy, sizeof(RBEzy));

	cudaMalloc((void**)&dev_CAEy, sizeof(CAEy));
	cudaMalloc((void**)&dev_CBEy, sizeof(CBEy));
	cudaMalloc((void**)&dev_RAExz, sizeof(RAExz));
	cudaMalloc((void**)&dev_RBExz, sizeof(RBExz));
	cudaMalloc((void**)&dev_RAEzx, sizeof(RAEzx));
	cudaMalloc((void**)&dev_RBEzx, sizeof(RBEzx));

	cudaMalloc((void**)&dev_CAEz, sizeof(CAEz));
	cudaMalloc((void**)&dev_CBEz, sizeof(CBEz));
	cudaMalloc((void**)&dev_RAExy, sizeof(RAExy));
	cudaMalloc((void**)&dev_RBExy, sizeof(RBExy));
	cudaMalloc((void**)&dev_RAEyx, sizeof(RAEyx));
	cudaMalloc((void**)&dev_RBEyx, sizeof(RBEyx));

	cudaMalloc((void**)&dev_CPHx, sizeof(CPHx));
	cudaMalloc((void**)&dev_CQHx, sizeof(CQHx));
	cudaMalloc((void**)&dev_RAHyz, sizeof(RAHyz));
	cudaMalloc((void**)&dev_RBHyz, sizeof(RBHyz));
	cudaMalloc((void**)&dev_RAHzy, sizeof(RAHzy));
	cudaMalloc((void**)&dev_RBHzy, sizeof(RBHzy));

	cudaMalloc((void**)&dev_CPHy, sizeof(CPHy));
	cudaMalloc((void**)&dev_CQHy, sizeof(CQHy));
	cudaMalloc((void**)&dev_RAHxz, sizeof(RAHxz));
	cudaMalloc((void**)&dev_RBHxz, sizeof(RBHxz));
	cudaMalloc((void**)&dev_RAHzx, sizeof(RAHzx));
	cudaMalloc((void**)&dev_RBHzx, sizeof(RBHzx));

	cudaMalloc((void**)&dev_CPHz, sizeof(CPHz));
	cudaMalloc((void**)&dev_CQHz, sizeof(CQHz));
	cudaMalloc((void**)&dev_RAHxy, sizeof(RAHxy));
	cudaMalloc((void**)&dev_RBHxy, sizeof(RBHxy));
	cudaMalloc((void**)&dev_RAHyx, sizeof(RAHyx));
	cudaMalloc((void**)&dev_RBHyx, sizeof(RBHyx));

	cudaMalloc((void**)&dev_kx_Ey, sizeof(kx_Ey));
	cudaMalloc((void**)&dev_kx_Ez, sizeof(kx_Ez));
	cudaMalloc((void**)&dev_ky_Ex, sizeof(ky_Ex));
	cudaMalloc((void**)&dev_ky_Ez, sizeof(ky_Ez));
	cudaMalloc((void**)&dev_kz_Ex, sizeof(kz_Ex));
	cudaMalloc((void**)&dev_kz_Ey, sizeof(kz_Ey));

	cudaMalloc((void**)&dev_kx_Hy, sizeof(kx_Hy));
	cudaMalloc((void**)&dev_kx_Hz, sizeof(kx_Hz));
	cudaMalloc((void**)&dev_ky_Hx, sizeof(ky_Hx));
	cudaMalloc((void**)&dev_ky_Hz, sizeof(ky_Hz));
	cudaMalloc((void**)&dev_kz_Hx, sizeof(kz_Hx));
	cudaMalloc((void**)&dev_kz_Hy, sizeof(kz_Hy));

	//gpu显存新创建数组，原来内存中不存在
	int szEx = nx * (ny + 1)*(nz + 1);
	int szEy = (nx + 1)*ny*(nz + 1);
	int szEz = (nx + 1)*(ny + 1)*nz;
	int szHx = (nx + 1)*ny*nz;
	int szHy = nx * (ny + 1)*nz;
	int szHz = nx * ny*(nz + 1);
	cudaMalloc((void**)&dev_Ex, szEx * sizeof(float));
	cudaMalloc((void**)&dev_UEyz, szEx * sizeof(float));
	cudaMalloc((void**)&dev_UEzy, szEx * sizeof(float));

	cudaMalloc((void**)&dev_Ey, szEy * sizeof(float));
	cudaMalloc((void**)&dev_UEzx, szEy * sizeof(float));
	cudaMalloc((void**)&dev_UExz, szEy * sizeof(float));

	cudaMalloc((void**)&dev_Ez, szEz * sizeof(float));
	cudaMalloc((void**)&dev_UExy, szEz * sizeof(float));
	cudaMalloc((void**)&dev_UEyx, szEz * sizeof(float));

	cudaMalloc((void**)&dev_Hx, szHx * sizeof(float));
	cudaMalloc((void**)&dev_UHyz, szHx * sizeof(float));
	cudaMalloc((void**)&dev_UHzy, szHx * sizeof(float));

	cudaMalloc((void**)&dev_Hy, szHy * sizeof(float));
	cudaMalloc((void**)&dev_UHzx, szHy * sizeof(float));
	cudaMalloc((void**)&dev_UHxz, szHy * sizeof(float));

	cudaMalloc((void**)&dev_Hz, szHz * sizeof(float));
	cudaMalloc((void**)&dev_UHxy, szHz * sizeof(float));
	cudaMalloc((void**)&dev_UHyx, szHz * sizeof(float));

	cudaMalloc((void**)&dev_V, sizeof(V));
	cudaMalloc((void**)&dev_E_obs, sizeof(E_obs));
	cudaMalloc((void**)&dev_source, sizeof(source));

	// 第二部分并行需要用到的变量
	cudaMalloc((void**)&dev_Ex_zheng_1, sizeof(Ex_zheng_1));
	cudaMalloc((void**)&dev_Ex_zheng_2, sizeof(Ex_zheng_2));
	cudaMalloc((void**)&dev_Ex_zheng_3, sizeof(Ex_zheng_3));

	cudaMalloc((void**)&dev_Ey_zheng_1, sizeof(Ey_zheng_1));
	cudaMalloc((void**)&dev_Ey_zheng_2, sizeof(Ey_zheng_2));
	cudaMalloc((void**)&dev_Ey_zheng_3, sizeof(Ey_zheng_3));

	cudaMalloc((void**)&dev_Ez_zheng_1, sizeof(Ez_zheng_1));
	cudaMalloc((void**)&dev_Ez_zheng_2, sizeof(Ez_zheng_2));
	cudaMalloc((void**)&dev_Ez_zheng_3, sizeof(Ez_zheng_3));

	cudaMalloc((void**)&dev_Hx_zheng_1, sizeof(Hx_zheng_1));
	cudaMalloc((void**)&dev_Hx_zheng_2, sizeof(Hx_zheng_2));
	cudaMalloc((void**)&dev_Hx_zheng_3, sizeof(Hx_zheng_3));

	cudaMalloc((void**)&dev_Hy_zheng_1, sizeof(Hy_zheng_1));
	cudaMalloc((void**)&dev_Hy_zheng_2, sizeof(Hy_zheng_2));
	cudaMalloc((void**)&dev_Hy_zheng_3, sizeof(Hy_zheng_3));

	cudaMalloc((void**)&dev_Hz_zheng_1, sizeof(Hz_zheng_1));
	cudaMalloc((void**)&dev_Hz_zheng_2, sizeof(Hz_zheng_2));
	cudaMalloc((void**)&dev_Hz_zheng_3, sizeof(Hz_zheng_3));

	cudaMalloc((void**)&dev_Ex_zheng_last, sizeof(Ex_zheng_last));
	cudaMalloc((void**)&dev_Ey_zheng_last, sizeof(Ey_zheng_last));
	cudaMalloc((void**)&dev_Ez_zheng_last, sizeof(Ez_zheng_last));
	cudaMalloc((void**)&dev_Hx_zheng_last, sizeof(Hx_zheng_last));
	cudaMalloc((void**)&dev_Hy_zheng_last, sizeof(Hy_zheng_last));
	cudaMalloc((void**)&dev_Hz_zheng_last, sizeof(Hz_zheng_last));

	cudaMalloc((void**)&dev_fan, sizeof(fan));
	cudaMalloc((void**)&dev_huanyuan,sizeof(huanyuan));

	cudaMalloc((void**)&dev_)
}


// flag == 0 将GPU显存中的E*, UE**, H*, UH**, (V, E_obs)置零
// flag == 1 将GPU显存中的E*, UE**, H*, UH**, (V, E*_zheng_*, H*_zheng_*, E*_zheng_last, H*_zheng_last, fan, huanyuan)置零
// flag == 2 将GPU显存中的E*, UE**, H*, UH**, (V, E*1, H*1, )置零
void gpu_memory_set_zero(int flag)
{
	int szEx = nx * (ny + 1)*(nz + 1);
	int szEy = (nx + 1)*ny*(nz + 1);
	int szEz = (nx + 1)*(ny + 1)*nz;
	int szHx = (nx + 1)*ny*nz;
	int szHy = nx * (ny + 1)*nz;
	int szHz = nx * ny*(nz + 1);


	//gpu显存新创建数组，原来内存中不存在
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

		cudaMemSet(dev_Ex_zheng_1, 0, sizeof(Ex_zheng_1));
		cudaMemSet(dev_Ex_zheng_2, 0, sizeof(Ex_zheng_2));
		cudaMemSet(dev_Ex_zheng_3, 0, sizeof(Ex_zheng_3));

		cudaMemSet(dev_Ey_zheng_1, 0, sizeof(Ey_zheng_1));
		cudaMemSet(dev_Ey_zheng_2, 0, sizeof(Ey_zheng_2));
		cudaMemSet(dev_Ey_zheng_3, 0, sizeof(Ey_zheng_3));

		cudaMemSet(dev_Ez_zheng_1, 0, sizeof(Ez_zheng_1));
		cudaMemSet(dev_Ez_zheng_2, 0, sizeof(Ez_zheng_2));
		cudaMemSet(dev_Ez_zheng_3, 0, sizeof(Ez_zheng_3));

		cudaMemSet(dev_Hx_zheng_1, 0, sizeof(Hx_zheng_1));
		cudaMemSet(dev_Hx_zheng_2, 0, sizeof(Hx_zheng_2));
		cudaMemSet(dev_Hx_zheng_3, 0, sizeof(Hx_zheng_3));

		cudaMemSet(dev_Hy_zheng_1, 0, sizeof(Hy_zheng_1));
		cudaMemSet(dev_Hy_zheng_2, 0, sizeof(Hy_zheng_2));
		cudaMemSet(dev_Hy_zheng_3, 0, sizeof(Hy_zheng_3));

		cudaMemSet(dev_Hz_zheng_1, 0, sizeof(Hz_zheng_1));
		cudaMemSet(dev_Hz_zheng_2, 0, sizeof(Hz_zheng_2));
		cudaMemSet(dev_Hz_zheng_3, 0, sizeof(Hz_zheng_3));

		cudaMemSet(dev_Ex_zheng_last, 0, sizeof(Ex_zheng_last));
		cudaMemSet(dev_Ey_zheng_last, 0, sizeof(Ey_zheng_last));
		cudaMemSet(dev_Ez_zheng_last, 0, sizeof(Ez_zheng_last));
		cudaMemSet(dev_Hx_zheng_last, 0, sizeof(Hx_zheng_last));
		cudaMemSet(dev_Hy_zheng_last, 0, sizeof(Hy_zheng_last));
		cudaMemSet(dev_Hz_zheng_last, 0, sizeof(Hz_zheng_last));

		cudaMemSet(dev_fan, 0, sizeof(fan));
		cudaMemSet(dev_huanyuan, 0, sizeof(huanyuan));
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

// 将内存中的CAE CBE RAE RBE CPH CQH RAH CBH k*_E* k*_H* source复制到显存中
void gpu_memory_copy()
{
	cudaError_t cudaStatus;
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_CAEx, CAEx, sizeof(CAEx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_CBEx, CBEx, sizeof(CBEx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAEyz, RAEyz, sizeof(RAEyz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBEyz, RBEyz, sizeof(RBEyz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAEzy, RAEzy, sizeof(RAEzy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBEzy, RBEzy, sizeof(RBEzy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMemcpy(dev_CAEy, CAEy, sizeof(CAEy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_CBEy, CBEy, sizeof(CBEy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAExz, RAExz, sizeof(RAExz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBExz, RBExz, sizeof(RBExz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAEzx, RAEzx, sizeof(RAEzx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBEzx, RBEzx, sizeof(RBEzx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMemcpy(dev_CAEz, CAEz, sizeof(CAEz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_CBEz, CBEz, sizeof(CBEz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAExy, RAExy, sizeof(RAExy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBExy, RBExy, sizeof(RBExy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAEyx, RAEyx, sizeof(RAEyx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBEyx, RBEyx, sizeof(RBEyx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMemcpy(dev_CPHx, CPHx, sizeof(CPHx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_CQHx, CQHx, sizeof(CQHx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAHyz, RAHyz, sizeof(RAHyz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBHyz, RBHyz, sizeof(RBHyz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAHzy, RAHzy, sizeof(RAHzy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBHzy, RBHzy, sizeof(RBHzy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMemcpy(dev_CPHy, CPHy, sizeof(CPHy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_CQHy, CQHy, sizeof(CQHy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAHxz, RAHxz, sizeof(RAHxz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBHxz, RBHxz, sizeof(RBHxz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAHzx, RAHzx, sizeof(RAHzx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBHzx, RBHzx, sizeof(RBHzx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMemcpy(dev_CPHz, CPHz, sizeof(CPHz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_CQHz, CQHz, sizeof(CQHz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAHxy, RAHxy, sizeof(RAHxy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBHxy, RBHxy, sizeof(RBHxy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RAHyx, RAHyx, sizeof(RAHyx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_RBHyx, RBHyx, sizeof(RBHyx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMemcpy(dev_kx_Ey, kx_Ey, sizeof(kx_Ey), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_kx_Ez, kx_Ez, sizeof(kx_Ez), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_ky_Ex, ky_Ex, sizeof(ky_Ex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_ky_Ez, ky_Ez, sizeof(ky_Ez), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_kz_Ex, kz_Ex, sizeof(kz_Ex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_kz_Ey, kz_Ey, sizeof(kz_Ey), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMemcpy(dev_kx_Hy, kx_Hy, sizeof(kx_Hy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_kx_Hz, kx_Hz, sizeof(kx_Hz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_ky_Hx, ky_Hx, sizeof(ky_Hx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_ky_Hz, ky_Hz, sizeof(ky_Hz), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_kz_Hx, kz_Hx, sizeof(kz_Hx), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_kz_Hy, kz_Hy, sizeof(kz_Hy), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMemcpy(dev_source, source, sizeof(source), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy failed!"); goto Error; }
Error:
	return;
}

void gpu_memory_free()
{
	cudaFree(dev_CAEx);
	cudaFree(dev_CBEx);
	cudaFree(dev_RAEyz);
	cudaFree(dev_RBEyz);
	cudaFree(dev_RAEzy);
	cudaFree(dev_RBEzy);

	cudaFree(dev_CAEy);
	cudaFree(dev_CBEy);
	cudaFree(dev_RAExz);
	cudaFree(dev_RBExz);
	cudaFree(dev_RAEzx);
	cudaFree(dev_RBEzx);

	cudaFree(dev_CAEz);
	cudaFree(dev_CBEz);
	cudaFree(dev_RAExy);
	cudaFree(dev_RBExy);
	cudaFree(dev_RAEyx);
	cudaFree(dev_RBEyx);

	cudaFree(dev_CPHx);
	cudaFree(dev_CQHx);
	cudaFree(dev_RAHyz);
	cudaFree(dev_RBHyz);
	cudaFree(dev_RAHzy);
	cudaFree(dev_RBHzy);

	cudaFree(dev_CPHy);
	cudaFree(dev_CQHy);
	cudaFree(dev_RAHxz);
	cudaFree(dev_RBHxz);
	cudaFree(dev_RAHzx);
	cudaFree(dev_RBHzx);

	cudaFree(dev_CPHz);
	cudaFree(dev_CQHz);
	cudaFree(dev_RAHxy);
	cudaFree(dev_RBHxy);
	cudaFree(dev_RAHyx);
	cudaFree(dev_RBHyx);

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

	cudaFree(dev_Ex);
	cudaFree(dev_UEyz);
	cudaFree(dev_UEzy);

	cudaFree(dev_Ey);
	cudaFree(dev_UEzx);
	cudaFree(dev_UExz);

	cudaFree(dev_Ez);
	cudaFree(dev_UExy);
	cudaFree(dev_UEyx);

	cudaFree(dev_Hx);
	cudaFree(dev_UHyz);
	cudaFree(dev_UHzy);

	cudaFree(dev_Hy);
	cudaFree(dev_UHzx);
	cudaFree(dev_UHxz);

	cudaFree(dev_Hz);
	cudaFree(dev_UHxy);
	cudaFree(dev_UHyx);

	cudaFree(dev_V);
	cudaFree(dev_E_obs);
	cudaFree(dev_source);
}

// 计算UH H UE E
cudaError_t zhengYan()
{
	cudaError_t cudaStatus;	
	calcUHyz << < gridUHyz, blockUHyz >> > (dev_UHyz, dev_RBHyz, dev_RAHyz, dev_Ez, dy);
	calcUHzy << < gridUHzy, blockUHzy >> > (dev_UHzy, dev_RBHzy, dev_RAHzy, dev_Ey, dz);
	calcUHxy << < gridUHxy, blockUHxy >> > (dev_UHxy, dev_RBHxy, dev_RAHxy, dev_Ey, dx);
	calcUHxz << < gridUHxz, blockUHxz >> > (dev_UHxz, dev_RBHxz, dev_RAHxz, dev_Ez, dx);
	calcUHyx << < gridUHyx, blockUHyx >> > (dev_UHyx, dev_RBHyx, dev_RAHyx, dev_Ex, dy);
	calcUHzx << < gridUHzx, blockUHzx >> > (dev_UHzx, dev_RBHzx, dev_RAHzx, dev_Ex, dz);

	calcHx << < gridHx, blockHx >> > (dev_Hx, dev_CPHx, dev_CQHx, dev_ky_Hx, dev_kz_Hx, dev_Ez, dev_Ey, dev_UHyz, dev_UHzy, dy, dz);
	calcHy << < gridHy, blockHy >> > (dev_Hy, dev_CPHy, dev_CQHy, dev_kz_Hy, dev_kx_Hy, dev_Ex, dev_Ez, dev_UHzx, dev_UHxz, dz, dx);
	calcHz << < gridHz, blockHz >> > (dev_Hz, dev_CPHz, dev_CQHz, dev_kx_Hz, dev_ky_Hz, dev_Ey, dev_Ex, dev_UHxy, dev_UHyx, dx, dy);

	calcUExy << < gridUExy, blockUExy >> > (dev_UExy, dev_RBExy, dev_RAExy, dev_Hy, dx);
	calcUExz << < gridUExz, blockUExz >> > (dev_UExz, dev_RBExz, dev_RAExz, dev_Hz, dx);
	calcUEyx << < gridUEyx, blockUEyx >> > (dev_UEyx, dev_RBEyx, dev_RAEyx, dev_Hx, dy);
	calcUEyz << < gridUEyz, blockUEyz >> > (dev_UEyz, dev_RBEyz, dev_RAEyz, dev_Hz, dy);
	calcUEzx << < gridUEzx, blockUEzx >> > (dev_UEzx, dev_RBEzx, dev_RAEzx, dev_Hx, dz);
	calcUEzy << < gridUEzy, blockUEzy >> > (dev_UEzy, dev_RBEzy, dev_RAEzy, dev_Hy, dz);

	calcEx << < gridEx, blockEx >> > (dev_Ex, dev_CAEx, dev_CBEx, dev_ky_Ex, dev_kz_Ex, dev_Hz, dev_Hy, dev_UEyz, dev_UEzy, dy, dz);
	calcEy << < gridEy, blockEy >> > (dev_Ey, dev_CAEy, dev_CBEy, dev_kz_Ey, dev_kx_Ey, dev_Hx, dev_Hz, dev_UEzx, dev_UExz, dz, dx);
	calcEz << < gridEz, blockEz >> > (dev_Ez, dev_CAEz, dev_CBEz, dev_kx_Ez, dev_ky_Ez, dev_Hy, dev_Hx, dev_UExy, dev_UEyx, dx, dy);

	// 计算过程是否出错?
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) { printf("Zhengyan Calc Failed: %s\n", cudaGetErrorString(cudaStatus)); return cudaStatus; }


}

cudaError_t gpu_parallel_one()
{
	cudaError_t cudaStatus;

	// 调用kernel函数计算。
	int i, j;
	for (i = 0; i < szfsw; i++)
	{
		// flag == 1代表后半部分的并行，将GPU显存中的E*, UE**, H*, UH**, (V, E*_zheng_*, H*_zheng_*, E*_zheng_last, H*_zheng_last, fan, huanyuan)置零
		gpu_memory_set_zero(1);
		for (j = 0; j < it; j++)
		{
			if (j % 200 == 0)
			{
				printf("i = %3d / %d,  j = %4d / %d\n", i, szfsw, j, it);
			}

			// 实现MATLAB中的Ex[fswzx[i] - 1][fswzy[i] - 1][fswzz[i] - 1] = source[j];
			int fidx = (fswzx[i] - 1)*(ny + 1)*(nz + 1) + (fswzy[i] - 1)*(nz + 1) + fswzz[i] - 1;
			cudaStatus = cudaMemcpy(&(dev_Ex[fidx]), &(dev_source[j]), sizeof(float), cudaMemcpyDeviceToDevice);

			// 实现MATLAB中的V(j)=Ex(jswzx(i), jswzy(i), jswzz(i));
			int jidx = (jswzx[i] - 1)*(ny + 1)*(nz + 1) + (jswzy[i] - 1)*(nz + 1) + jswzz[i] - 1;
			cudaStatus = cudaMemcpy(&(dev_V[j]), &(dev_Ex[jidx]), sizeof(float), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) { printf("V cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus)); return cudaStatus; };

			cudaStatus = cudaMemcpy(&(E_obs[j][i]), &(dev_V[j]), sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) { printf("V cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus)); return cudaStatus; };
		}
	}

	printf("finish calc !\n");

	cudaDeviceSynchronize();

	return cudaStatus;
}

cudaError_t gpu_parallel_two()
{
	cudaError_t cudaStatus;

	// 调用kernel函数计算。
	int i, j;
	for (i = 0; i < szfsw; i++)
	{
		// flag == 1代表后半部分的并行，将GPU显存中的E*, UE**, H*, UH**, (V, E*_zheng_*, H*_zheng_*, E*_zheng_last, H*_zheng_last, fan, huanyuan)置零
		gpu_memory_set_zero(1);
		for (j = 0; j < it; j++)
		{
			if (j % 200 == 0)
			{
				printf("i = %3d / %d,  j = %4d / %d\n", i, szfsw, j, it);
			}

			// 实现MATLAB中的Ex[fswzx[i] - 1][fswzy[i] - 1][fswzz[i] - 1] = source[j];
			int fidx = (fswzx[i] - 1)*(ny + 1)*(nz + 1) + (fswzy[i] - 1)*(nz + 1) + fswzz[i] - 1;
			cudaStatus = cudaMemcpy(&(dev_Ex[fidx]), &(dev_source[j]), sizeof(float), cudaMemcpyDeviceToDevice);

			// 实现MATLAB中的V(j)=Ex(jswzx(i), jswzy(i), jswzz(i));
			int jidx = (jswzx[i] - 1)*(ny + 1)*(nz + 1) + (jswzy[i] - 1)*(nz + 1) + jswzz[i] - 1;
			cudaStatus = cudaMemcpy(&(dev_V[j]), &(dev_Ex[jidx]), sizeof(float), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) { printf("V cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus)); return cudaStatus; };

			cudaStatus = cudaMemcpy(&(E_obs[j][i]), &(dev_V[j]), sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) { printf("V cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus)); return cudaStatus; };
		}
	}

	printf("finish calc !\n");

	cudaDeviceSynchronize();

	return cudaStatus;
}

/************************************************************************************
* 主函数
************************************************************************************/
int main()
{
	printf("c = %e\n", dt);
	readAllData();
	printf("Read All Data OK ! \n");

	// 选择运算使用的GPU
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) { printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); return 1; }
	else { printf("cudaSetDevice success!\n"); }

	gpu_memory_malloc();
	gpu_memory_copy();

	// 调用gpu运算
	cudaStatus = calcWithCuda();
	if (cudaStatus != cudaSuccess) { printf("calcWithCuda failed!"); return 1; }
	else { printf("calcWithCudasuccess!\n"); }

	gpu_memory_free();

	// 重置GPU
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) { printf("cudaDeviceReset failed!"); return 1; }

	// 输出结果
	printE_obs();

	return 0;
}