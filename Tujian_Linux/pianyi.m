
%3D Cartesian coordinates wave forwarding code by Chongmin Zhang
%CPML 

clear all;
close all;
clc;
tic;

% parpool('SpmdEnabled',false)
% parpool open lo
load('E_obs.mat');

%*********************************************************************** 
% 锟斤拷锟?锟斤�?%***********************************************************************
c=single(2.99792458e8);             
mu_0=single(4.0*pi*1.0e-7);       
eps_0=single(1.0/(c*c*mu_0));      
freq=single(600);                                               
freq=single(freq*1.0e+6);                           
%********************************************************************
%锟斤拷锟斤拷锟斤�?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
%********************************************************************
dx=single(0.01);                                              
dy=dx;
dz=dx;
dt=single(0.8/(sqrt(1/dx^2+1/dy^2+1/dz^2)*c));   
% it=single(round(19*1e-9/dt));
it=400;
%********************************************************************
%PML锟斤拷锟斤拷
%********************************************************************
npml=single(10);
%锟斤拷锟斤拷呓锟斤拷�?npmlc=single(4);
%********************************************************************
%锟斤拷锟绞诧拷锟斤拷
%********************************************************************
index=single(ones(170,120,90));
param=[9 0.00001 1 0];%锟斤拷锟斤拷锟斤�?
[nx,ny,nz]=size(index);
nx=single(nx);
ny=single(ny);
nz=single(nz);

param1=single(reshape(param(index,1),size(index)));
param2=single(reshape(param(index,2),size(index)));
param3=single(reshape(param(index,3),size(index)));
param4=single(reshape(param(index,4),size(index)));



for i=1:it
     A=10;
     t=i*dt-1/freq;
     source(i)=A*(1-2*(pi*freq*t)^2)*exp(-(pi*freq*t)^2);
end

figure;
plot(source);
getframe(gca);


%===========锟斤拷锟斤拷锟斤拷锟轿伙拷锟?==============================

step=single(8);
dancxds=(npml+1):step:(nx-npml-3);
dancxds=length(dancxds);
cxsl=5;

fsx=(npml+1):step:(nx-npml-3);
jsx=(npml+3):step:(nx-npml-1);
fswzx=repmat(fsx,1,cxsl);
jswzx=repmat(jsx,1,cxsl);

step1=single(10);
fswzy=(npml+30):step1:(npml+70); 
jswzy=(npml+30):step1:(npml+70); 
fswzy=repelem(fswzy,dancxds);
jswzy=repelem(jswzy,dancxds);

fswzz=(nz-npml-1).*ones(1,dancxds*cxsl);
jswzz=(nz-npml-1).*ones(1,dancxds*cxsl);
%===================================================


m=single(4);
kmax=single(1);
alphamax=single(0);
bj=single(0.8);

%Ex nx,ny+1,nz+1
eps_r=single(zeros(nx,ny+1,nz+1));
sig  =single(zeros(nx,ny+1,nz+1));
eps_r(:,2:ny,2:nz)=single(1/4.*(param1(:,1:ny-1,1:nz-1)+param1(:,2:ny,1:nz-1)+param1(:,1:ny-1,2:nz)+param1(:,2:ny,2:nz)));
sig(:,2:ny,2:nz)=single(1/4.*(param2(:,1:ny-1,1:nz-1)+param2(:,2:ny,1:nz-1)+param2(:,1:ny-1,2:nz)+param2(:,2:ny,2:nz)));
eps=eps_r.*eps_0;

CAEx=single((2.*eps-sig.*dt)./(2.*eps+sig.*dt));
CBEx=single(2.*dt./(2.*eps+sig.*dt));

%UEyz=zeros(nx,ny+1,nz+1);
sigmax=bj.*(m+1)./(150.*pi.*dy)./sqrt(eps_r(:,[2:npml ny-npml+2:ny],2:nz));
sigy=sigmax.*repmat(reshape((([npml-1:-1:1 1:npml-1])./npml).^m,[1,2*(npml-1),1]),[nx,1,nz-1]);
ky_Ex=single(ones(nx,ny+1,nz+1));
ky=1+(kmax-1).*(([npml-1:-1:1 1:npml-1])./npml).^m;
ky=repmat(reshape(ky,[1,2*npml-2,1]),[nx,1,nz-1]);
ky_Ex(:,[2:npml ny-npml+2:ny],2:nz)=ky;
alphay=alphamax.*([npml-1:-1:1 1:npml-1]./npml);
alphay=repmat(reshape(alphay,[1,2*npml-2,1]),[nx,1,nz-1]);
RBEyz=single(exp(-((sigy./ky)+alphay).*(dt./eps_0)));
RAEyz=single(sigy./(sigy.*ky+ky.^2.*alphay).*(RBEyz-1));
clear sigmax sigy ky alphay
%UEzy=zeros(nx,ny+1,nz+1);
sigmax=bj.*(m+1)./(150.*pi.*dz)./sqrt(eps_r(:,2:ny,[2:npml nz-npml+2:nz]));
sigz=sigmax.*repmat(reshape(([npml-1:-1:1 1:npml-1]./npml).^m,[1,1,2*npml-2]),[nx,ny-1,1]);
kz_Ex=single(ones(nx,ny+1,nz+1));
kz=1+(kmax-1).*([npml-1:-1:1 1:npml-1]./npml).^m;
kz=repmat(reshape(kz,[1,1,2*npml-2]),[nx,ny-1,1]);
kz_Ex(:,2:ny,[2:npml nz-npml+2:nz])=kz;
alphaz=alphamax.*([npml-1:-1:1 1:npml-1]./npml);
alphaz=repmat(reshape(alphaz,[1,1,2*npml-2]),[nx,ny-1,1]);
RBEzy=single(exp(-((sigz./kz)+alphaz).*(dt./eps_0)));
RAEzy=single(sigz./(sigz.*kz+kz.^2.*alphaz).*(RBEzy-1));
clear eps_r eps sig sigmax sigz kz alphaz

%Ey nx+1,ny,nz+1
eps_r=single(zeros(nx+1,ny,nz+1));
sig  =single(zeros(nx+1,ny,nz+1));
eps_r(2:nx,:,2:nz)=single(1/4.*(param1(1:nx-1,:,1:nz-1)+param1(2:nx,:,1:nz-1)+param1(1:nx-1,:,2:nz)+param1(2:nx,:,2:nz)));
sig(2:nx,:,2:nz)=single(1/4.*(param2(1:nx-1,:,1:nz-1)+param2(2:nx,:,1:nz-1)+param2(1:nx-1,:,2:nz)+param2(2:nx,:,2:nz)));
eps=eps_r.*eps_0;

CAEy=single((2.*eps-sig.*dt)./(2.*eps+sig.*dt));
CBEy=single(2.*dt./(2.*eps+sig.*dt));

%UEzx=zeros(nx+1,ny,nz+1);
sigmax=bj.*(m+1)./(150.*pi.*dz)./sqrt(eps_r(2:nx,:,[2:npml nz-npml+2:nz]));
sigz=sigmax.*repmat(reshape(([npml-1:-1:1 1:npml-1]./npml).^m,[1,1,2*npml-2]),[nx-1,ny,1]);
kz_Ey=single(ones(nx+1,ny,nz+1));
kz=1+(kmax-1).*([npml-1:-1:1 1:npml-1]./npml).^m;
kz=repmat(reshape(kz,[1,1,2*npml-2]),[nx-1,ny,1]);
kz_Ey(2:nx,:,[2:npml nz-npml+2:nz])=kz;
alphaz=alphamax.*([npml-1:-1:1 1:npml-1]./npml);
alphaz=repmat(reshape(alphaz,[1,1,2*npml-2]),[nx-1,ny,1]);
RBEzx=single(exp(-((sigz./kz)+alphaz).*(dt./eps_0)));
RAEzx=single(sigz./(sigz.*kz+kz.^2.*alphaz).*(RBEzx-1));
clear sigmax sigz kz alphaz
%UExz=zeros(nx+1,ny,nz+1);
sigmax=bj.*(m+1)./(150.*pi.*dx)./sqrt(eps_r([2:npml nx-npml+2:nx],:,2:nz));
sigx=sigmax.*repmat(reshape((([npml-1:-1:1 1:npml-1])./npml).^m,[2*(npml-1),1,1]),[1,ny,nz-1]);
kx_Ey=single(ones(nx+1,ny,nz+1));
kx=1+(kmax-1).*(([npml-1:-1:1 1:npml-1]')./npml).^m;
kx=repmat(kx,[1,ny,nz-1]);
kx_Ey([2:npml nx-npml+2:nx],:,2:nz)=kx;
alphax=alphamax.*(([npml-1:-1:1 1:npml-1]')./npml);
alphax=reshape(alphax',[2*(npml-1),1,1]);
alphax=repmat(alphax,[1,ny,nz-1]);
RBExz=single(exp(-((sigx./kx)+alphax).*(dt./eps_0)));
RAExz=single(sigx./(sigx.*kx+kx.^2.*alphax).*(RBExz-1));
clear eps_r eps sig sigmax sigx kx alphax

%Ez nx+1,ny+1,nz
eps_r=single(zeros(nx+1,ny+1,nz));
sig  =single(zeros(nx+1,ny+1,nz));
eps_r(2:nx,2:ny,:)=single(1/4.*(param1(1:nx-1,1:ny-1,:)+param1(2:nx,1:ny-1,:)+param1(1:nx-1,2:ny,:)+param1(2:nx,2:ny,:)));
sig(2:nx,2:ny,:)=single(1/4.*(param2(1:nx-1,1:ny-1,:)+param2(2:nx,1:ny-1,:)+param2(1:nx-1,2:ny,:)+param2(2:nx,2:ny,:)));
eps=eps_r.*eps_0;

CAEz=single((2.*eps-sig.*dt)./(2.*eps+sig.*dt));
CBEz=single(2.*dt./(2.*eps+sig.*dt));
%UExy=zeros(nx+1,ny+1,nz);
sigmax=bj.*(m+1)./(150.*pi.*dx)./sqrt(eps_r([2:npml nx-npml+2:nx],2:ny,:));
sigx=sigmax.*repmat(reshape((([npml-1:-1:1 1:npml-1])./npml).^m,[2*(npml-1),1,1]),[1,ny-1,nz]);
kx_Ez=single(ones(nx+1,ny+1,nz));
kx=1+(kmax-1).*([npml-1:-1:1 1:npml-1]./npml).^m;
kx=repmat(reshape(kx,[2*npml-2,1,1]),[1,ny-1,nz]);
kx_Ez([2:npml nx-npml+2:nx],2:ny,:)=kx;
alphax=alphamax.*([npml-1:-1:1 1:npml-1]./npml);
alphax=repmat(reshape(alphax,[2*npml-2,1,1]),[1,ny-1,nz]);
RBExy=single(exp(-((sigx./kx)+alphax).*(dt./eps_0)));
RAExy=single(sigx./(sigx.*kx+kx.^2.*alphax).*(RBExy-1));
clear sigmax sigx kx alphax
%UEyx=zeros(nx+1,ny+1,nz);
sigmax=bj.*(m+1)./(150.*pi.*dy)./sqrt(eps_r(2:nx,[2:npml ny-npml+2:ny],:));
sigy=sigmax.*repmat(reshape((([npml-1:-1:1 1:npml-1])./npml).^m,[1,2*(npml-1),1]),[nx-1,1,nz]);
ky_Ez=single(ones(nx+1,ny+1,nz));
ky=1+(kmax-1).*([npml-1:-1:1 1:npml-1]./npml).^m;
ky=repmat(reshape(ky,[1,2*npml-2,1]),[nx-1,1,nz]);
ky_Ez(2:nx,[2:npml ny-npml+2:ny],:)=ky;
alphay=alphamax.*([npml-1:-1:1 1:npml-1]./npml);
alphay=repmat(reshape(alphay,[1,2*npml-2,1]),[nx-1,1,nz]);
RBEyx=single(exp(-((sigy./ky)+alphay).*(dt./eps_0)));
RAEyx=single(sigy./(sigy.*ky+ky.^2.*alphay).*(RBEyx-1));
clear eps_r eps sig sigmax sigy ky alphay

%Hx nx+1,ny,nz
eps_r=single(zeros(nx+1,ny,nz));
mu_r=single(zeros(nx+1,ny,nz));
sigm=single(zeros(nx+1,ny,nz));
eps_r(2:nx,:,:)=single(1/2.*(param1(1:nx-1,:,:)+param1(2:nx,:,:)));
mu_r(2:nx,:,:)=single(1/2.*(param3(1:nx-1,:,:)+param3(2:nx,:,:)));
sigm(2:nx,:,:)=single(1/2.*(param4(1:nx-1,:,:)+param4(2:nx,:,:)));
mu=mu_r.*mu_0;

CPHx=single((2.*mu-sigm.*dt)./(2.*mu+sigm.*dt));
CQHx=single(2.*dt./(2.*mu+sigm.*dt));

%UHyz=zeros(nx+1,ny,nz);
sigmax=bj.*(m+1)./(150.*pi.*dy)./sqrt(eps_r(2:nx,[1:npml ny-npml+1:ny],:));
sigy=sigmax.*repmat(reshape(([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m,[1,2*npml,1]),[nx-1,1,nz]);
ky_Hx=single(ones(nx+1,ny,nz));
ky=1+(kmax-1).*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m;
ky=repmat(reshape(ky,[1,2*npml,1]),[nx-1,1,nz]);
ky_Hx(2:nx,[1:npml ny-npml+1:ny],:)=ky;
alphay=alphamax.*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml);
alphay=repmat(reshape(alphay,[1,2*npml,1]),[nx-1,1,nz]);
RBHyz=single(exp(-((sigy./ky)+alphay).*(dt./eps_0)));
RAHyz=single(sigy./(sigy.*ky+ky.^2.*alphay).*(RBHyz-1));
clear sigmax sigy ky alphay
%UHzy=zeros(nx+1,ny,nz);
sigmax=bj.*(m+1)./(150.*pi.*dz)./sqrt(eps_r(2:nx,:,[1:npml nz-npml+1:nz]));
sigz=sigmax.*repmat(reshape(([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m,[1,1,2*npml]),[nx-1,ny,1]);
kz_Hx=single(ones(nx+1,ny,nz));
kz=1+(kmax-1).*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m;
kz=repmat(reshape(kz,[1,1,2*npml]),[nx-1,ny,1]);
kz_Hx(2:nx,:,[1:npml nz-npml+1:nz])=kz;
alphaz=alphamax.*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml);
alphaz=repmat(reshape(alphaz,[1,1,2*npml]),[nx-1,ny,1]);
RBHzy=single(exp(-((sigz./kz)+alphaz).*(dt./eps_0)));
RAHzy=single(sigz./(sigz.*kz+kz.^2.*alphaz).*(RBHzy-1));
clear eps_r mu_r mu sigm sigmax sigz kz alphaz

%Hy=zeros(nx,ny+1,nz);
eps_r=single(zeros(nx,ny+1,nz));
mu_r=single(zeros(nx,ny+1,nz));
sigm=single(zeros(nx,ny+1,nz));
eps_r(:,2:ny,:)=single(1/2.*(param1(:,1:ny-1,:)+param1(:,2:ny,:)));
mu_r(:,2:ny,:)=single(1/2.*(param3(:,1:ny-1,:)+param3(:,2:ny,:)));
sigm(:,2:ny,:)=single(1/2.*(param4(:,1:ny-1,:)+param4(:,2:ny,:)));
mu=mu_r.*mu_0;

CPHy=single((2.*mu-sigm.*dt)./(2.*mu+sigm.*dt));
CQHy=single(2.*dt./(2.*mu+sigm.*dt));

%UHzx=zeros(nx,ny+1,nz)
sigmax=bj.*(m+1)./(150.*pi.*dz)./sqrt(eps_r(:,2:ny,[1:npml nz-npml+1:nz]));
sigz=sigmax.*repmat(reshape(([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m,[1,1,2*npml]),[nx,ny-1,1]);
kz_Hy=single(ones(nx,ny+1,nz));
kz=1+(kmax-1).*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m;
kz=repmat(reshape(kz,[1,1,2*npml]),[nx,ny-1,1]);
kz_Hy(:,2:ny,[1:npml nz-npml+1:nz])=kz;
alphaz=alphamax.*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml);
alphaz=repmat(reshape(alphaz,[1,1,2*npml]),[nx,ny-1,1]);
RBHzx=single(exp(-((sigz./kz)+alphaz).*(dt./eps_0)));
RAHzx=single(sigz./(sigz.*kz+kz.^2.*alphaz).*(RBHzx-1));
clear sigmax sigz kz alphaz
%UHxz=zeros(nx,ny+1,nz);
sigmax=bj.*(m+1)./(150.*pi.*dx)./sqrt(eps_r([1:npml nx-npml+1:nx],2:ny,:));
sigx=sigmax.*repmat(reshape(([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m,[2*npml,1,1]),[1,ny-1,nz]);
kx_Hy=single(ones(nx,ny+1,nz));
kx=1+(kmax-1).*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m;
kx=repmat(reshape(kx,[2*npml,1,1]),[1,ny-1,nz]);
kz_Hy([1:npml nx-npml+1:nx],2:ny,:)=kx;
alphax=alphamax.*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml);
alphax=repmat(reshape(alphax,[2*npml,1,1]),[1,ny-1,nz]);
RBHxz=single(exp(-((sigx./kx)+alphax).*(dt./eps_0)));
RAHxz=single(sigx./(sigx.*kx+kx.^2.*alphax).*(RBHxz-1));
clear eps_r mu_r mu sigm sigmax sigx kx alphax

%Hz=zeros(nx,ny,nz+1);
eps_r=single(zeros(nx,ny,nz+1));
mu_r=single(zeros(nx,ny,nz+1));
sigm=single(zeros(nx,ny,nz+1));
eps_r(:,:,2:nz)=single(1/2.*(param1(:,:,1:nz-1)+param1(:,:,2:nz)));
mu_r(:,:,2:nz)=single(1/2.*(param3(:,:,1:nz-1)+param3(:,:,2:nz)));
sigm(:,:,2:nz)=single(1/2.*(param4(:,:,1:nz-1)+param4(:,:,2:nz)));
mu=mu_r.*mu_0;

CPHz=single((2.*mu-sigm.*dt)./(2.*mu+sigm.*dt));
CQHz=single(2.*dt./(2.*mu+sigm.*dt));

%UHxy=zeros(nx,ny,nz+1);
sigmax=bj.*(m+1)./(150.*pi.*dx)./sqrt(eps_r([1:npml nx-npml+1:nx],:,2:nz));
sigx=sigmax.*repmat(reshape(([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m,[2*npml,1,1]),[1,ny,nz-1]);
kx_Hz=single(ones(nx,ny,nz+1));
kx=1+(kmax-1).*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m;
kx=repmat(reshape(kx,[2*npml,1,1]),[1,ny,nz-1]);
kx_Hz([1:npml nx-npml+1:nx],:,2:nz)=kx;
alphax=alphamax.*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml);
alphax=repmat(reshape(alphax,[2*npml,1,1]),[1,ny,nz-1]);
RBHxy=single(exp(-((sigx./kx)+alphax).*(dt./eps_0)));
RAHxy=single(sigx./(sigx.*kx+kx.^2.*alphax).*(RBHxy-1));
clear sigmax sigx kx alphax
%UHyx=zeros(nx,ny,nz+1);
sigmax=bj.*(m+1)./(150.*pi.*dy)./sqrt(eps_r(:,[1:npml ny-npml+1:ny],2:nz));
sigy=sigmax.*repmat(reshape(([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m,[1,2*npml,1]),[nx,1,nz-1]);
ky_Hz=single(ones(nx,ny,nz+1));
ky=1+(kmax-1).*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml).^m;
ky=repmat(reshape(ky,[1,2*npml,1]),[nx,1,nz-1]);
ky_Hz(:,[1:npml ny-npml+1:ny],2:nz)=ky;
alphay=alphamax.*([npml-0.5:-1:0.5 0.5:npml-0.5]./npml);
alphay=repmat(reshape(alphay,[1,2*npml,1]),[nx,1,nz-1]);
RBHyx=single(exp(-((sigy./ky)+alphay).*(dt./eps_0)));
RAHyx=single(sigy./(sigy.*ky+ky.^2.*alphay).*(RBHyx-1));
clear eps_r mu_r mu sigm sigmax sigy ky alphay
