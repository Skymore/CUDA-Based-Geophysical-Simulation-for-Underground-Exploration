#include<math.h>
#include<stdlib.h>

#define it (1233)
#define npml (10)
#define npmlc (4)
#define np (14) // np = npml + npmlc;
#define nx (170)
#define ny (120)
#define nz (90)
#define szfsw (370)
#define dx (0.01)
#define dy (0.01)
#define dz (0.01)
#define pi (3.141592653589)
#define path "/home/stu/taorui0117/Tujian_Linux" // 程序运行的目录
#define cudaDevice 0 // 程序使用的gpu
#define isPianYi true
/************************************************************************************
* 内存参数表
************************************************************************************/
const float c = 2.99792458e8;
const float mu_0 = 4.0*pi*1.0e-7;
const float eps_0 = 1.0 / (c*c*mu_0);

const float dt = 1 / (sqrt(1 / (dx*dx)+1 / (dy*dy)+1 / (dz*dz))*c);
const float freq = 600*1.0e6;

float Ex[nx][ny+1][nz+1];
float Ey[nx+1][ny][nz+1];
float Ez[nx+1][ny+1][nz];

float UEyz[nx][ny+1][nz+1];
float UEzy[nx][ny+1][nz+1];
float UExz[nx+1][ny][nz+1];
float UEzx[nx+1][ny][nz+1];
float UExy[nx+1][ny+1][nz];
float UEyx[nx+1][ny+1][nz];

float Hx[nx+1][ny][nz];
float Hy[nx][ny+1][nz];
float Hz[nx][ny][nz+1];

float UHyz[nx+1][ny][nz];
float UHzy[nx+1][ny][nz];
float UHxz[nx][ny+1][nz];
float UHzx[nx][ny+1][nz];
float UHxy[nx][ny][nz+1];
float UHyx[nx][ny][nz+1];

float CAEx[nx][ny+1][nz+1];
float CAEy[nx+1][ny][nz+1];
float CAEz[nx+1][ny+1][nz];

float CBEx[nx][ny+1][nz+1];
float CBEy[nx+1][ny][nz+1];
float CBEz[nx+1][ny+1][nz];

float RAEyz[nx][2*(npml-1)][nz-1];
float RAEzy[nx][ny-1][2*(npml-1)];
float RAEzx[nx-1][ny][2*(npml-1)];
float RAExz[2*(npml-1)][ny][nz-1];
float RAExy[2*(npml-1)][ny-1][nz];
float RAEyx[nx-1][2*(npml-1)][nz];

float RBEyz[nx][2*(npml-1)][nz-1];
float RBEzy[nx][ny-1][2*(npml-1)];
float RBEzx[nx-1][ny][2*(npml-1)];
float RBExz[2*(npml-1)][ny][nz-1];
float RBExy[2*(npml-1)][ny-1][nz];
float RBEyx[nx-1][2*(npml-1)][nz];

float CPHx[nx+1][ny][nz];
float CQHx[nx+1][ny][nz];
float CPHy[nx][ny+1][nz];
float CQHy[nx][ny+1][nz];
float CPHz[nx][ny][nz+1];
float CQHz[nx][ny][nz+1];

float RAHyz[nx-1][2*npml][nz];
float RAHzy[nx-1][ny][2*npml];
float RAHzx[nx][ny-1][2*npml];
float RAHxz[2*npml][ny-1][nz];
float RAHxy[2*npml][ny][nz-1];
float RAHyx[nx][2*npml][nz-1];

float RBHyz[nx-1][2*npml][nz];
float RBHzy[nx-1][ny][2*npml];
float RBHzx[nx][ny-1][2*npml];
float RBHxz[2*npml][ny-1][nz];
float RBHxy[2*npml][ny][nz-1];
float RBHyx[nx][2*npml][nz-1];


int fswzx[szfsw];
int fswzy[szfsw];
int fswzz[szfsw];
int jswzx[szfsw];
int jswzy[szfsw];
int jswzz[szfsw];

float E_obs[it][szfsw];
float V[it];
float source[it];


float kx_Ey[nx+1][ny][nz+1];
float kx_Ez[nx+1][ny+1][nz];
float ky_Ex[nx][ny+1][nz+1];
float ky_Ez[nx+1][ny+1][nz];
float kz_Ex[nx][ny+1][nz+1];
float kz_Ey[nx+1][ny][nz+1];

float kx_Hy[nx][ny+1][nz];
float kx_Hz[nx][ny][nz+1];
float ky_Hx[nx+1][ny][nz];
float ky_Hz[nx][ny][nz+1];
float kz_Hx[nx+1][ny][nz];
float kz_Hy[nx][ny+1][nz];

/*
float Ex_zheng_1[it][2*npml][ny-2*npml][nz-2*npml];
float Ex_zheng_2[it][nx-2*npml][2*npml][nz-2*npml];
float Ex_zheng_3[it][nx-2*npml][ny-2*npml][2*npml];

float Ey_zheng_1[it][2*npml][ny-2*npml][nz-2*npml];
float Ey_zheng_2[it][nx-2*npml][2*npml][nz-2*npml];
float Ey_zheng_3[it][nx-2*npml][ny-2*npml][2*npml];

float Ez_zheng_1[it][2*npml][ny-2*npml][nz-2*npml];
float Ez_zheng_2[it][nx-2*npml][2*npml][nz-2*npml];
float Ez_zheng_3[it][nx-2*npml][ny-2*npml][2*npml];

float Hx_zheng_1[it][2*npml][ny-2*npml][nz-2*npml];
float Hx_zheng_2[it][nx-2*npml][2*npml][nz-2*npml];
float Hx_zheng_3[it][nx-2*npml][ny-2*npml][2*npml];

float Hy_zheng_1[it][2*npml][ny-2*npml][nz-2*npml];
float Hy_zheng_2[it][nx-2*npml][2*npml][nz-2*npml];
float Hy_zheng_3[it][nx-2*npml][ny-2*npml][2*npml];

float Hz_zheng_1[it][2*npml][ny-2*npml][nz-2*npml];
float Hz_zheng_2[it][nx-2*npml][2*npml][nz-2*npml];
float Hz_zheng_3[it][nx-2*npml][ny-2*npml][2*npml];

float Ex_zheng_last[nx-2*npml][ny-2*npml][nz-2*npml];
float Ey_zheng_last[nx-2*npml][ny-2*npml][nz-2*npml];
float Ez_zheng_last[nx-2*npml][ny-2*npml][nz-2*npml];

float Hx_zheng_last[nx-2*npml][ny-2*npml][nz-2*npml];
float Hy_zheng_last[nx-2*npml][ny-2*npml][nz-2*npml];
float Hz_zheng_last[nx-2*npml][ny-2*npml][nz-2*npml];
*/

float *Ex_zheng_1 = (float*)malloc((it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
float *Ex_zheng_2 = (float*)malloc((it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
float *Ex_zheng_3 = (float*)malloc((it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

float *Ey_zheng_1 = (float*)malloc((it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
float *Ey_zheng_2 = (float*)malloc((it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
float *Ey_zheng_3 = (float*)malloc((it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

float *Ez_zheng_1 = (float*)malloc((it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
float *Ez_zheng_2 = (float*)malloc((it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
float *Ez_zheng_3 = (float*)malloc((it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

float *Hx_zheng_1 = (float*)malloc((it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
float *Hx_zheng_2 = (float*)malloc((it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
float *Hx_zheng_3 = (float*)malloc((it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

float *Hy_zheng_1 = (float*)malloc((it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
float *Hy_zheng_2 = (float*)malloc((it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
float *Hy_zheng_3 = (float*)malloc((it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

float *Hz_zheng_1 = (float*)malloc((it)*(2 * npmlc)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
float *Hz_zheng_2 = (float*)malloc((it)*(nx - 2 * npml)*(2 * npmlc)*(nz - 2 * npml) * sizeof(float));
float *Hz_zheng_3 = (float*)malloc((it)*(nx - 2 * npml)*(ny - 2 * npml)*(2 * npmlc) * sizeof(float));

float *Ex_zheng_last = (float*)malloc((nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
float *Ey_zheng_last = (float*)malloc((nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
float *Ez_zheng_last = (float*)malloc((nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));

float *Hx_zheng_last = (float*)malloc((nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
float *Hy_zheng_last = (float*)malloc((nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));
float *Hz_zheng_last = (float*)malloc((nx - 2 * npml)*(ny - 2 * npml)*(nz - 2 * npml) * sizeof(float));

float fan[nx-2*npml][ny-2*npml][nz-2*npml];
float huanyuan[nx-2*npml][ny-2*npml][nz-2*npml];

float Ex1[nx][ny+1][nz+1];
float Ey1[nx+1][ny][nz+1];
float Ez1[nx+1][ny+1][nz];

float Hx1[nx+1][ny][nz];
float Hy1[nx][ny+1][nz];
float Hz1[nx][ny][nz+1];

float ns[nx - 2 * npml][ny - 2 * npml][nz - 2 * npml];
float zv[nx - 2 * npml][ny - 2 * npml][nz - 2 * npml];
float fv[nx - 2 * npml][ny - 2 * npml][nz - 2 * npml];
/************************************************************************************
* 显存数组
************************************************************************************/
float *dev_Ex;
float *dev_Ey;
float *dev_Ez;

float *dev_UEyz;
float *dev_UEzy;
float *dev_UExz;
float *dev_UEzx;
float *dev_UExy;
float *dev_UEyx;

float *dev_Hx;
float *dev_Hy;
float *dev_Hz;

float *dev_UHyz;
float *dev_UHzy;
float *dev_UHxz;
float *dev_UHzx;
float *dev_UHxy;
float *dev_UHyx;

float *dev_CAEx;
float *dev_CAEy;
float *dev_CAEz;

float *dev_CBEx;
float *dev_CBEy;
float *dev_CBEz;

float *dev_RAEyz;
float *dev_RAEzy;
float *dev_RAEzx;
float *dev_RAExz;
float *dev_RAExy;
float *dev_RAEyx;

float *dev_RBEyz;
float *dev_RBEzy;
float *dev_RBEzx;
float *dev_RBExz;
float *dev_RBExy;
float *dev_RBEyx;

float *dev_CPHx;
float *dev_CQHx;
float *dev_CPHy;
float *dev_CQHy;
float *dev_CPHz;
float *dev_CQHz;

float *dev_RAHyz;
float *dev_RAHzy;
float *dev_RAHzx;
float *dev_RAHxz;
float *dev_RAHxy;
float *dev_RAHyx;

float *dev_RBHyz;
float *dev_RBHzy;
float *dev_RBHzx;
float *dev_RBHxz;
float *dev_RBHxy;
float *dev_RBHyx;


int *dev_fswzx;
int *dev_fswzy;
int *dev_fswzz;
int *dev_jswzx;
int *dev_jswzy;
int *dev_jswzz;

float *dev_E_obs;
float *dev_V;
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

float *dev_Ex_zheng_1;
float *dev_Ex_zheng_2;
float *dev_Ex_zheng_3;

float *dev_Ey_zheng_1;
float *dev_Ey_zheng_2;
float *dev_Ey_zheng_3;

float *dev_Ez_zheng_1;
float *dev_Ez_zheng_2;
float *dev_Ez_zheng_3;

float *dev_Hx_zheng_1;
float *dev_Hx_zheng_2;
float *dev_Hx_zheng_3;

float *dev_Hy_zheng_1;
float *dev_Hy_zheng_2;
float *dev_Hy_zheng_3;

float *dev_Hz_zheng_1;
float *dev_Hz_zheng_2;
float *dev_Hz_zheng_3;

float *dev_Ex_zheng_last;
float *dev_Ey_zheng_last;
float *dev_Ez_zheng_last;

float *dev_Hx_zheng_last;
float *dev_Hy_zheng_last;
float *dev_Hz_zheng_last;

float *dev_fan;
float *dev_huanyuan;

float *dev_Ex1;
float *dev_Ey1;
float *dev_Ez1;

float *dev_Hx1;
float *dev_Hy1;
float *dev_Hz1;

float *dev_ns;
float *dev_zv;
float *dev_fv;

void freeMemory()
{
	free(Ex_zheng_1);
	free(Ex_zheng_2);
	free(Ex_zheng_3);

	free(Ey_zheng_1);
	free(Ey_zheng_2);
	free(Ey_zheng_3);

	free(Ez_zheng_1);
	free(Ez_zheng_2);
	free(Ez_zheng_3);

	free(Hx_zheng_1);
	free(Hx_zheng_2);
	free(Hx_zheng_3);

	free(Hy_zheng_1);
	free(Hy_zheng_2);
	free(Hy_zheng_3);

	free(Hz_zheng_1);
	free(Hz_zheng_2);
	free(Hz_zheng_3);

	free(Ex_zheng_last);
	free(Ey_zheng_last);
	free(Ez_zheng_last);

	free(Hx_zheng_last);
	free(Hy_zheng_last);
	free(Hz_zheng_last);
}