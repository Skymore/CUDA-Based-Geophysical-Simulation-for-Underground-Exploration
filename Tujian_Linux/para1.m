E_obs=single(zeros(it,length(fswzx)));

parfor i=single(1):single(length(fswzx))

Ex=zeros(nx,ny+1,nz+1);
UEyz=zeros(nx,ny+1,nz+1);
UEzy=zeros(nx,ny+1,nz+1);
Ey=zeros(nx+1,ny,nz+1);
UEzx=zeros(nx+1,ny,nz+1);
UExz=zeros(nx+1,ny,nz+1);
Ez=zeros(nx+1,ny+1,nz);
UExy=zeros(nx+1,ny+1,nz);
UEyx=zeros(nx+1,ny+1,nz);
Hx=zeros(nx+1,ny,nz);
UHyz=zeros(nx+1,ny,nz);
UHzy=zeros(nx+1,ny,nz);
Hy=zeros(nx,ny+1,nz);
UHzx=zeros(nx,ny+1,nz);
UHxz=zeros(nx,ny+1,nz);
Hz=zeros(nx,ny,nz+1);
UHxy=zeros(nx,ny,nz+1);
UHyx=zeros(nx,ny,nz+1);
V=zeros(it,1);


	for j=single(1):single(it)
        
		if mod(j,100)==0
			disp([num2str(i) '/' num2str(length(fswzx)) '	' num2str(j) '/' num2str(it)]);
		end
        
        Ex(fswzx(i),fswzy(i),fswzz(i))=source(j);
		 
        UHyz(2:nx,[1:npml ny-npml+1:ny],:)=RBHyz.*UHyz(2:nx,[1:npml ny-npml+1:ny],:)...
            +RAHyz./dy.*(Ez(2:nx,[2:npml+1 ny-npml+2:ny+1],:)-Ez(2:nx,[1:npml ny-npml+1:ny],:));
		UHzy(2:nx,:,[1:npml nz-npml+1:nz])=RBHzy.*UHzy(2:nx,:,[1:npml nz-npml+1:nz])...
            +RAHzy./dz.*(Ey(2:nx,:,[2:npml+1 nz-npml+2:nz+1])-Ey(2:nx,:,[1:npml nz-npml+1:nz]));
		Hx(2:nx,:,:)=CPHx(2:nx,:,:).*Hx(2:nx,:,:)-CQHx(2:nx,:,:)./ky_Hx(2:nx,:,:)./dy.*(Ez(2:nx,2:ny+1,:)-Ez(2:nx,1:ny,:))+CQHx(2:nx,:,:)./kz_Hx(2:nx,:,:)./dz.*(Ey(2:nx,:,2:nz+1)-Ey(2:nx,:,1:nz))...
            -CQHx(2:nx,:,:).*UHyz(2:nx,:,:)+CQHx(2:nx,:,:).*UHzy(2:nx,:,:);
       

		UHzx(:,2:ny,[1:npml nz-npml+1:nz])=RBHzx.*UHzx(:,2:ny,[1:npml nz-npml+1:nz])+...
            RAHzx./dz.*(Ex(:,2:ny,[2:npml+1 nz-npml+2:nz+1])-Ex(:,2:ny,[1:npml nz-npml+1:nz]));
	    UHxz([1:npml nx-npml+1:nx],2:ny,:)=RBHxz.*UHxz([1:npml nx-npml+1:nx],2:ny,:)...
            +RAHxz./dx.*(Ez([2:npml+1 nx-npml+2:nx+1],2:ny,:)-Ez([1:npml nx-npml+1:nx],2:ny,:));
        Hy(:,2:ny,:)=CPHy(:,2:ny,:).*Hy(:,2:ny,:)-CQHy(:,2:ny,:)./kz_Hy(:,2:ny,:)./dz.*(Ex(:,2:ny,2:nz+1)-Ex(:,2:ny,1:nz))+CQHy(:,2:ny,:)./kx_Hy(:,2:ny,:)./dx.*(Ez(2:nx+1,2:ny,:)-Ez(1:nx,2:ny,:))...
            -CQHy(:,2:ny,:).*UHzx(:,2:ny,:)+CQHy(:,2:ny,:).*UHxz(:,2:ny,:);
        
		UHxy([1:npml nx-npml+1:nx],:,2:nz)=RBHxy.*UHxy([1:npml nx-npml+1:nx],:,2:nz)...
            +RAHxy./dx.*(Ey([2:npml+1 nx-npml+2:nx+1],:,2:nz)-Ey([1:npml nx-npml+1:nx],:,2:nz));	
		UHyx(:,[1:npml ny-npml+1:ny],2:nz)=RBHyx.*UHyx(:,[1:npml ny-npml+1:ny],2:nz)...
            +RAHyx./dy.*(Ex(:,[2:npml+1 ny-npml+2:ny+1],2:nz)-Ex(:,[1:npml ny-npml+1:ny],2:nz));	
	    Hz(:,:,2:nz)=CPHz(:,:,2:nz).*Hz(:,:,2:nz)-CQHz(:,:,2:nz)./kx_Hz(:,:,2:nz)./dx.*(Ey(2:nx+1,:,2:nz)-Ey(1:nx,:,2:nz))+CQHz(:,:,2:nz)./ky_Hz(:,:,2:nz)./dy.*(Ex(:,2:ny+1,2:nz)-Ex(:,1:ny,2:nz))...
            -CQHz(:,:,2:nz).*UHxy(:,:,2:nz)+CQHz(:,:,2:nz).*UHyx(:,:,2:nz); 
        
        UEyz(:,[2:npml ny-npml+2:ny],2:nz)=RBEyz.*UEyz(:,[2:npml ny-npml+2:ny],2:nz)...
            +RAEyz./dy.*(Hz(:,[2:npml ny-npml+2:ny],2:nz)-Hz(:,[1:npml-1 ny-npml+1:ny-1],2:nz));
		UEzy(:,2:ny,[2:npml nz-npml+2:nz])=RBEzy.*UEzy(:,2:ny,[2:npml nz-npml+2:nz])...
            +RAEzy./dz.*(Hy(:,2:ny,[2:npml nz-npml+2:nz])-Hy(:,2:ny,[1:npml-1 nz-npml+1:nz-1]));
		Ex(:,2:ny,2:nz)=CAEx(:,2:ny,2:nz).*Ex(:,2:ny,2:nz)...
            +CBEx(:,2:ny,2:nz)./ky_Ex(:,2:ny,2:nz)./dy.*(Hz(:,2:ny,2:nz)-Hz(:,1:ny-1,2:nz))...
            -CBEx(:,2:ny,2:nz)./kz_Ex(:,2:ny,2:nz)./dz.*(Hy(:,2:ny,2:nz)-Hy(:,2:ny,1:nz-1))...
            +CBEx(:,2:ny,2:nz).*UEyz(:,2:ny,2:nz)-CBEx(:,2:ny,2:nz).*UEzy(:,2:ny,2:nz);
		
		UEzx(2:nx,:,[2:npml nz-npml+2:nz])=RBEzx.*UEzx(2:nx,:,[2:npml nz-npml+2:nz])...
            +RAEzx./dz.*(Hx(2:nx,:,[2:npml nz-npml+2:nz])-Hx(2:nx,:,[1:npml-1 nz-npml+1:nz-1]));
        UExz([2:npml nx-npml+2:nx],:,2:nz)=RBExz.*UExz([2:npml nx-npml+2:nx],:,2:nz)...
            +RAExz./dx.*(Hz([2:npml nx-npml+2:nx],:,2:nz)-Hz([1:npml-1 nx-npml+1:nx-1],:,2:nz));
        Ey(2:nx,:,2:nz)=CAEy(2:nx,:,2:nz).*Ey(2:nx,:,2:nz)...
            +CBEy(2:nx,:,2:nz)./kz_Ey(2:nx,:,2:nz)./dz.*(Hx(2:nx,:,2:nz)-Hx(2:nx,:,1:nz-1))...
            -CBEy(2:nx,:,2:nz)./kx_Ey(2:nx,:,2:nz)./dx.*(Hz(2:nx,:,2:nz)-Hz(1:nx-1,:,2:nz))...
            +CBEy(2:nx,:,2:nz).*UEzx(2:nx,:,2:nz)-CBEy(2:nx,:,2:nz).*UExz(2:nx,:,2:nz);   	

        UExy([2:npml nx-npml+2:nx],2:ny,:)=RBExy.*UExy([2:npml nx-npml+2:nx],2:ny,:)...
            +RAExy./dx.*(Hy([2:npml nx-npml+2:nx],2:ny,:)-Hy([1:npml-1 nx-npml+1:nx-1],2:ny,:));  
		UEyx(2:nx,[2:npml ny-npml+2:ny],:)=RBEyx.*UEyx(2:nx,[2:npml ny-npml+2:ny],:)...
            +RAEyx./dy.*(Hx(2:nx,[2:npml ny-npml+2:ny],:)-Hx(2:nx,[1:npml-1 ny-npml+1:ny-1],:));
        Ez(2:nx,2:ny,:)=CAEz(2:nx,2:ny,:).*Ez(2:nx,2:ny,:)...
            +CBEz(2:nx,2:ny,:)./kx_Ez(2:nx,2:ny,:)./dx.*(Hy(2:nx,2:ny,:)-Hy(1:nx-1,2:ny,:))...
            -CBEz(2:nx,2:ny,:)./kx_Ez(2:nx,2:ny,:)./dy.*(Hx(2:nx,2:ny,:)-Hx(2:nx,1:ny-1,:))...
            +CBEz(2:nx,2:ny,:).*UExy(2:nx,2:ny,:)-CBEz(2:nx,2:ny,:).*UEyx(2:nx,2:ny,:);   

        V(j)=Ex(jswzx(i),jswzy(i),jswzz(i));
        
    end
    
    E_obs(:,i)=V;
end

save('E_obs.mat')