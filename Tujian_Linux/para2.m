npmlc = 4;
%=======================================================
ns=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));
zv=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));
fv=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));

for i=single(1):single(length(fswzx))
    Ex_zheng_1=single(zeros(2*npmlc,ny-2*npml,nz-2*npml,it));
    Ex_zheng_2=single(zeros(nx-2*npml,2*npmlc,nz-2*npml,it));
    Ex_zheng_3=single(zeros(nx-2*npml,ny-2*npml,2*npmlc,it));
    
    Ey_zheng_1=single(zeros(2*npmlc,ny-2*npml,nz-2*npml,it));
    Ey_zheng_2=single(zeros(nx-2*npml,2*npmlc,nz-2*npml,it));
    Ey_zheng_3=single(zeros(nx-2*npml,ny-2*npml,2*npmlc,it));
    
    Ez_zheng_1=single(zeros(2*npmlc,ny-2*npml,nz-2*npml,it));
    Ez_zheng_2=single(zeros(nx-2*npml,2*npmlc,nz-2*npml,it));
    Ez_zheng_3=single(zeros(nx-2*npml,ny-2*npml,2*npmlc,it));
    
    Hx_zheng_1=single(zeros(2*npmlc,ny-2*npml,nz-2*npml,it));
    Hx_zheng_2=single(zeros(nx-2*npml,2*npmlc,nz-2*npml,it)); 
    Hx_zheng_3=single(zeros(nx-2*npml,ny-2*npml,2*npmlc,it));
    
    Hy_zheng_1=single(zeros(2*npmlc,ny-2*npml,nz-2*npml,it));
    Hy_zheng_2=single(zeros(nx-2*npml,2*npmlc,nz-2*npml,it));
    Hy_zheng_3=single(zeros(nx-2*npml,ny-2*npml,2*npmlc,it));
    
    Hz_zheng_1=single(zeros(2*npmlc,ny-2*npml,nz-2*npml,it));
    Hz_zheng_2=single(zeros(nx-2*npml,2*npmlc,nz-2*npml,it));
    Hz_zheng_3=single(zeros(nx-2*npml,ny-2*npml,2*npmlc,it));

    Ex_zheng_last=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));
    Ey_zheng_last=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));
    Ez_zheng_last=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));
    Hx_zheng_last=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));
    Hy_zheng_last=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));
    Hz_zheng_last=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));
    
    fan=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));
    huanyuan=single(zeros(nx-2*npml,ny-2*npml,nz-2*npml));

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
            disp([num2str(i) '/' num2str(length(fswzx)) '    ' num2str(j) '/' num2str(it)]);
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

        Ex_zheng_1(:,:,:,j)=Ex([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml);
        Ey_zheng_1(:,:,:,j)=Ey([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml);
        Ez_zheng_1(:,:,:,j)=Ez([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml);
        Hx_zheng_1(:,:,:,j)=Hx([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml);
        Hy_zheng_1(:,:,:,j)=Hy([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml);
        Hz_zheng_1(:,:,:,j)=Hz([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml);

        Ex_zheng_2(:,:,:,j)=Ex(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml);
        Ey_zheng_2(:,:,:,j)=Ey(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml);
        Ez_zheng_2(:,:,:,j)=Ez(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml);
        Hx_zheng_2(:,:,:,j)=Hx(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml);
        Hz_zheng_2(:,:,:,j)=Hz(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml);
        Hy_zheng_2(:,:,:,j)=Hy(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml);

        Ex_zheng_3(:,:,:,j)=Ex(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml]);    
        Ey_zheng_3(:,:,:,j)=Ey(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml]);  
        Ez_zheng_3(:,:,:,j)=Ez(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml]);
        Hx_zheng_3(:,:,:,j)=Hx(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml]);    
        Hy_zheng_3(:,:,:,j)=Hy(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml]);    
        Hz_zheng_3(:,:,:,j)=Hz(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml]);    
        
        Ex_zheng_last=Ex(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
        Ey_zheng_last=Ey(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
        Ez_zheng_last=Ez(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
        Hx_zheng_last=Hx(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
        Hy_zheng_last=Hy(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
        Hz_zheng_last=Hz(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
        
    end
    
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
    Ex1=zeros(nx,ny+1,nz+1);
    Ey1=zeros(nx+1,ny,nz+1);
    Ez1=zeros(nx+1,ny+1,nz);
    Hx1=zeros(nx+1,ny,nz);
    Hy1=zeros(nx,ny+1,nz);
    Hz1=zeros(nx,ny,nz+1);

    for j=single(it):-1:single(1)
        
        if mod(j,100)==0
            disp([num2str(i+2*length(fswzx)) '/' num2str(length(fswzx)*3) '    ' num2str(j) '/' num2str(it)]);
        end
        
        Ex(fswzx(i),fswzy(i),fswzz(i))=E_obs(j,i);
         
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
            -CBEz(2:nx,2:ny,:)./kx_Ez(2:nx,2:ny,:)./dy.*(Hx(2:nx,2:ny,:)-Hx(2:nx,1:ny-1,:));

        fan=Ex(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
        
        if j==it
            Ex1(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml)=Ex_zheng_last;
            Ey1(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml)=Ey_zheng_last;
            Ez1(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml)=Ez_zheng_last;
            Hx1(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml)=Hx_zheng_last;
            Hy1(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml)=Hy_zheng_last;
            Hz1(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml)=Hz_zheng_last;
        else
        
            Hx1([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml)=Hx_zheng_1(:,:,:,j);
            Hx1(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml)=Hx_zheng_2(:,:,:,j);
            Hx1(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml])=Hx_zheng_3(:,:,:,j);
            
            Hy1([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml)=Hy_zheng_1(:,:,:,j);
            Hy1(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml)=Hy_zheng_2(:,:,:,j);
            Hy1(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml])=Hy_zheng_3(:,:,:,j);

            Hz1([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml)=Hz_zheng_1(:,:,:,j);
            Hz1(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml)=Hz_zheng_2(:,:,:,j);
            Hz1(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml])=Hz_zheng_3(:,:,:,j);
            
            Ex1([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml)=Ex_zheng_1(:,:,:,j);
            Ex1(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml)=Ex_zheng_2(:,:,:,j);
            Ex1(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml])=Ex_zheng_3(:,:,:,j);
            
            Ey1([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml)=Ey_zheng_1(:,:,:,j);
            Ey1(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml)=Ey_zheng_2(:,:,:,j);
            Ey1(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml])=Ey_zheng_3(:,:,:,j);
            
            Ez1([npml+1:npml+npmlc nx-npml-npmlc+1:nx-npml],npml+1:ny-npml,npml+1:nz-npml)=Ez_zheng_1(:,:,:,j);
            Ez1(npml+1:nx-npml,[npml+1:npml+npmlc ny-npml-npmlc+1:ny-npml],npml+1:nz-npml)=Ez_zheng_2(:,:,:,j);
            Ez1(npml+1:nx-npml,npml+1:ny-npml,[npml+1:npml+npmlc nz-npml-npmlc+1:nz-npml])=Ez_zheng_3(:,:,:,j);
            
            Ex1(fswzx(i),fswzy(i),fswzz(i))=source(j);
            
            Hx1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)=1./CPHx(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*Hx1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)...
            +1./CPHx(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CQHx(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dy.*(Ez1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1+1:ny-npml-npmlc+1,npml+npmlc+1:nz-npml-npmlc)-Ez1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc))...    
            -1./CPHx(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CQHx(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dz.*(Ey1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1+1:nz-npml-npmlc+1)-Ey1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc));
                
            Hy1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)=1./CPHy(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*Hy1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)...
            +1./CPHy(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CQHy(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dz.*(Ex1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1+1:nz-npml-npmlc+1)-Ex1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc))...
            -1./CPHy(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CQHy(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dx.*(Ez1(npml+npmlc+1+1:nx-npml-npmlc+1,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)-Ez1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc));
            
            Hz1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)=1./CPHz(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*Hz1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)...
            +1./CPHz(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CQHz(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dx.*(Ey1(npml+npmlc+1+1:nx-npml-npmlc+1,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)-Ey1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc))...
            -1./CPHz(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CQHz(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dy.*(Ex1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1+1:ny-npml-npmlc+1,npml+npmlc+1:nz-npml-npmlc)-Ex1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)); 

            Ex1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)=1./CAEx(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*Ex1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)...
                -1./CAEx(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CBEx(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dy.*(Hz1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)-Hz1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc:ny-npml-npmlc-1,npml+npmlc+1:nz-npml-npmlc))...
                +1./CAEx(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CBEx(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dz.*(Hy1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)-Hy1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc:nz-npml-npmlc-1));
                
            Ey1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)=1./CAEy(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*Ey1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)...
                -1./CAEy(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CBEy(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dz.*(Hx1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)-Hx1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc:nz-npml-npmlc-1))...
                +1./CAEy(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CBEy(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dx.*(Hz1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)-Hz1(npml+npmlc:nx-npml-npmlc-1,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc));
             
            Ez1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)=1./CAEz(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*Ez1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)...
                -1./CAEz(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CBEz(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dx.*(Hy1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)-Hy1(npml+npmlc:nx-npml-npmlc-1,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc))...
                +1./CAEz(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc).*CBEz(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)./dy.*(Hx1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc+1:ny-npml-npmlc,npml+npmlc+1:nz-npml-npmlc)-Hx1(npml+npmlc+1:nx-npml-npmlc,npml+npmlc:ny-npml-npmlc-1,npml+npmlc+1:nz-npml-npmlc));
          end
        %fan=Ex(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
        huanyuan=Ex1(npml+1:nx-npml,npml+1:ny-npml,npml+1:nz-npml);
%         
%       ns=ns+huanyuan.*fan;
%       zv=zv+huanyuan.*huanyuan;
%         fv=ns./zv;
        
        ns=ns+mean(huanyuan.*fan,4);
        zv=zv+mean(huanyuan.*huanyuan,4);
        fv=fv+mean(fan.*fan,4);
        
        %‘⁄cuda¿Ô∞¥’‚—˘À„µƒ
        %ns=ns+(huanyuan.*fan);
        %zv=zv+(huanyuan.*huanyuan);
        %fv=fv+(fan.*fan);
        
    end % end of for j=single(it):-1:single(1)
    test = 0;
end % end of for i.....


save('ns.mat');
save('zv.mat');
save('fv.mat');

figure
set(gcf,'outerposition',get(0,'screensize'))
imagesc(dz.*(1:nz-2*npml),dy.*(1:ny-2*npml),squeeze(ns(nx./2,:,:)))
colorbar
colormap(jet)
set(gca,'FontName','Times New Roman','FontSize',36)
xlabel('\fontname{???§Ê?∑È???§Ê??}?¢È???§Ê?∑È???§Ê?∑È??Áµ?fontname{Times New Roman}/m')
ylabel('\fontname{???§Ê?∑È???§Ê??}?¢È???§Ê?∑È???§Ê?∑È??Áµ?fontname{Times New Roman}/m')
getframe(gcf);
figure
set(gcf,'outerposition',get(0,'screensize'))
imagesc(dz.*(1:nz-2*npml),dy.*(1:ny-2*npml),squeeze(fv(nx./2,:,:)))
colorbar
colormap(jet)
set(gca,'FontName','Times New Roman','FontSize',36)
xlabel('\fontname{???§Ê?∑È???§Ê??}?¢È???§Ê?∑È???§Ê?∑È??Áµ?fontname{Times New Roman}/m')
ylabel('\fontname{???§Ê?∑È???§Ê??}?¢È???§Ê?∑È???§Ê?∑È??Áµ?fontname{Times New Roman}/m')
getframe(gcf);


figure
set(gcf,'outerposition',get(0,'screensize'))
imagesc(dz.*(1:nz-2*npml),dy.*(1:ny-2*npml),squeeze(ns(nx./2,:,:)./zv(nx./2,:,:)))
colorbar
colormap(jet)
set(gca,'FontName','Times New Roman','FontSize',36)
xlabel('\fontname{???§Ê?∑È???§Ê??}?¢È???§Ê?∑È???§Ê?∑È??Áµ?fontname{Times New Roman}/m')
ylabel('\fontname{???§Ê?∑È???§Ê??}?¢È???§Ê?∑È???§Ê?∑È??Áµ?fontname{Times New Roman}/m')
getframe(gcf);

figure
set(gcf,'outerposition',get(0,'screensize'))
imagesc(dz.*(1:nz-2*npml),dy.*(1:ny-2*npml),squeeze(ns(nx./2,:,:)./fv(nx./2,:,:)))
colorbar
colormap(jet)
set(gca,'FontName','Times New Roman','FontSize',36)
xlabel('\fontname{???§Ê?∑È???§Ê??}?¢È???§Ê?∑È???§Ê?∑È??Áµ?fontname{Times New Roman}/m')
ylabel('\fontname{???§Ê?∑È???§Ê??}?¢È???§Ê?∑È???§Ê?∑È??Áµ?fontname{Times New Roman}/m')
getframe(gcf);

figure
set(gcf,'outerposition',get(0,'screensize'))
imagesc(dz.*(1:nz-2*npml),dy.*(1:ny-2*npml),squeeze(ns(nx./2,:,:)./sqrt(zv(nx./2,:,:).*fv(nx./2,:,:))))
colorbar
colormap(jet)
set(gca,'FontName','Times New Roman','FontSize',36)
xlabel('\fontname{???§Ê?∑È???§Ê??}?¢È???§Ê?∑È???§Ê?∑È??Áµ?fontname{Times New Roman}/m')
ylabel('\fontname{???§Ê?∑È???§Ê??}?¢È???§Ê?∑È???§Ê?∑È??Áµ?fontname{Times New Roman}/m')
getframe(gcf);


toc;