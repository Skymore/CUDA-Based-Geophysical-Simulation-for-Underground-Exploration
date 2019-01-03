#include<math.h>

#define it (1000)
#define npml (10)
#define nx (170)
#define ny (120)
#define nz (120)
#define szfsw (185)
/************************************************************************************
* 内存参数表
************************************************************************************/
const float pi = 3.14159265358979323846;
const float c = 2.99792458e8;
const float mu_0 = 4.0*pi*1.0e-7;
const float eps_0 = 1.0 / (c*c*mu_0);
const float dx = 0.01;
const float dy = 0.01;
const float dz = 0.01;
const float dt = 1 / (sqrt(1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz))*c);
const float freq = 500 * 1.0e6;


float Ex[nx][ny + 1][nz + 1], UEyz[nx][ny + 1][nz + 1], UEzy[nx][ny + 1][nz + 1];
float Ey[nx + 1][ny][nz + 1], UExz[nx + 1][ny][nz + 1], UEzx[nx + 1][ny][nz + 1];
float Ez[nx + 1][ny + 1][nz], UExy[nx + 1][ny + 1][nz], UEyx[nx + 1][ny + 1][nz];

float Hx[nx + 1][ny][nz], UHyz[nx + 1][ny][nz], UHzy[nx + 1][ny][nz];
float Hy[nx][ny + 1][nz], UHxz[nx][ny + 1][nz], UHzx[nx][ny + 1][nz];
float Hz[nx][ny][nz + 1], UHxy[nx][ny][nz + 1], UHyx[nx][ny][nz + 1];

float CAEx[nx][ny + 1][nz + 1];
float CBEx[nx][ny + 1][nz + 1];
float RAEyz[nx][2 * (npml - 1)][nz - 1];
float RBEyz[nx][2 * (npml - 1)][nz - 1];
float RAEzy[nx][ny - 1][2 * (npml - 1)];
float RBEzy[nx][ny - 1][2 * (npml - 1)];
float CAEy[nx + 1][ny][nz + 1];
float CBEy[nx + 1][ny][nz + 1];
float RAEzx[nx - 1][ny][2 * (npml - 1)];
float RBEzx[nx - 1][ny][2 * (npml - 1)];
float RAExz[2 * (npml - 1)][ny][nz - 1];
float RBExz[2 * (npml - 1)][ny][nz - 1];
float CAEz[nx + 1][ny + 1][nz];
float CBEz[nx + 1][ny + 1][nz];
float RAExy[2 * (npml - 1)][ny - 1][nz];
float RBExy[2 * (npml - 1)][ny - 1][nz];
float RAEyx[nx - 1][2 * (npml - 1)][nz];
float RBEyx[nx - 1][2 * (npml - 1)][nz];
float CPHx[nx + 1][ny][nz];
float CQHx[nx + 1][ny][nz];
float RAHyz[nx - 1][2 * npml][nz];
float RBHyz[nx - 1][2 * npml][nz];
float RAHzy[nx - 1][ny][2 * npml];
float RBHzy[nx - 1][ny][2 * npml];
float CPHy[nx][ny + 1][nz];
float CQHy[nx][ny + 1][nz];
float RAHzx[nx][ny - 1][2 * npml];
float RBHzx[nx][ny - 1][2 * npml];
float RAHxz[2 * npml][ny - 1][nz];
float RBHxz[2 * npml][ny - 1][nz];
float CPHz[nx][ny][nz + 1];
float CQHz[nx][ny][nz + 1];
float RAHxy[2 * npml][ny][nz - 1];
float RBHxy[2 * npml][ny][nz - 1];
float RAHyx[nx][2 * npml][nz - 1];
float RBHyx[nx][2 * npml][nz - 1];
float source[it];
int fswzx[szfsw];
int fswzy[szfsw];
int fswzz[szfsw];
int jswzx[szfsw];
int jswzy[szfsw];
int jswzz[szfsw];
float E_obs[it][szfsw];
float V[it];
float kx_Ey[nx + 1][ny][nz + 1];
float kx_Ez[nx + 1][ny + 1][nz];
float ky_Ex[nx][ny + 1][nz + 1];
float ky_Ez[nx + 1][ny + 1][nz];
float kz_Ex[nx][ny + 1][nz + 1];
float kz_Ey[nx + 1][ny][nz + 1];

float kx_Hy[nx][ny + 1][nz];
float kx_Hz[nx][ny][nz + 1];
float ky_Hx[nx + 1][ny][nz];
float ky_Hz[nx][ny][nz + 1];
float kz_Hx[nx + 1][ny][nz];
float kz_Hy[nx][ny + 1][nz];

/************************************************************************************
* 显存数组 传参数定义
************************************************************************************/
float *dev_CAEx;
float *dev_CBEx;
float *dev_RAEyz;
float *dev_RBEyz;
float *dev_RAEzy;
float *dev_RBEzy;
float *dev_CAEy;
float *dev_CBEy;
float *dev_RAEzx;
float *dev_RBEzx;
float *dev_RAExz;
float *dev_RBExz;
float *dev_CAEz;
float *dev_CBEz;
float *dev_RAExy;
float *dev_RBExy;
float *dev_RAEyx;
float *dev_RBEyx;
float *dev_CPHx;
float *dev_CQHx;
float *dev_RAHyz;
float *dev_RBHyz;
float *dev_RAHzy;
float *dev_RBHzy;
float *dev_CPHy;
float *dev_CQHy;
float *dev_RAHzx;
float *dev_RBHzx;
float *dev_RAHxz;
float *dev_RBHxz;
float *dev_CPHz;
float *dev_CQHz;
float *dev_RAHxy;
float *dev_RBHxy;
float *dev_RAHyx;
float *dev_RBHyx;
float *dev_source;

float *dev_kx_Ey;
float *dev_kx_Ez;
float *dev_ky_Ex;
float *dev_ky_Ez;
float *dev_kz_Ex;
float *dev_kz_Ey;

float *dev_kx_Hy;
float *dev_kx_Hz;
float *dev_ky_Hx;
float *dev_ky_Hz;
float *dev_kz_Hx;
float *dev_kz_Hy;


/************************************************************************************
* 显存数组 新定义
************************************************************************************/
float *dev_Ex, *dev_UEyz, *dev_UEzy;
float *dev_Ey, *dev_UExz, *dev_UEzx;
float *dev_Ez, *dev_UExy, *dev_UEyx;

float *dev_Hx, *dev_UHyz, *dev_UHzy;
float *dev_Hy, *dev_UHxz, *dev_UHzx;
float *dev_Hz, *dev_UHxy, *dev_UHyx;
float *dev_V;
float *dev_E_obs;