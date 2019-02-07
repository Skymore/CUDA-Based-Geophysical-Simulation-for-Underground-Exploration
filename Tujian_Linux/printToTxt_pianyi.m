tic;
path = '/Users/taorui/Documents/GitHub/Tujian_github/Tujian_Linux/data_pianyi';
dlmwrite([path,'/CAEx.txt'],CAEx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/CAEy.txt'],CAEy,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/CAEz.txt'],CAEz,'delimiter','\t','precision',10, 'newline', 'pc');

dlmwrite([path,'/CBEx.txt'],CBEx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/CBEy.txt'],CBEy,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/CBEz.txt'],CBEz,'delimiter','\t','precision',10, 'newline', 'pc');

dlmwrite([path,'/CPHx.txt'],CPHx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/CPHy.txt'],CPHy,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/CPHz.txt'],CPHz,'delimiter','\t','precision',10, 'newline', 'pc');

dlmwrite([path,'/CQHx.txt'],CQHx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/CQHy.txt'],CQHy,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/CQHz.txt'],CQHz,'delimiter','\t','precision',10, 'newline', 'pc');

%RAE
dlmwrite([path,'/RAExy.txt'],RAExy,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RAExz.txt'],RAExz,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RAEyx.txt'],RAEyx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RAEyz.txt'],RAEyz,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RAEzx.txt'],RAEzx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RAEzy.txt'],RAEzy,'delimiter','\t','precision',10, 'newline', 'pc');

%RAH
dlmwrite([path,'/RAHxy.txt'],RAHxy,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RAHxz.txt'],RAHxz,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RAHyx.txt'],RAHyx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RAHyz.txt'],RAHyz,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RAHzx.txt'],RAHzx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RAHzy.txt'],RAHzy,'delimiter','\t','precision',10, 'newline', 'pc');

%RBE
dlmwrite([path,'/RBExy.txt'],RBExy,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RBExz.txt'],RBExz,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RBEyx.txt'],RBEyx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RBEyz.txt'],RBEyz,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RBEzx.txt'],RBEzx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RBEzy.txt'],RBEzy,'delimiter','\t','precision',10, 'newline', 'pc');

%RBH
dlmwrite([path,'/RBHxy.txt'],RBHxy,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RBHxz.txt'],RBHxz,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RBHyx.txt'],RBHyx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RBHyz.txt'],RBHyz,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RBHzx.txt'],RBHzx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/RBHzy.txt'],RBHzy,'delimiter','\t','precision',10, 'newline', 'pc');

%fsw
dlmwrite([path,'/fswzx.txt'],fswzx,'delimiter','\t','precision','%d', 'newline', 'pc');
dlmwrite([path,'/fswzy.txt'],fswzy,'delimiter','\t','precision','%d', 'newline', 'pc');
dlmwrite([path,'/fswzz.txt'],fswzz,'delimiter','\t','precision','%d', 'newline', 'pc');

dlmwrite([path,'/jswzx.txt'],jswzx,'delimiter','\t','precision','%d', 'newline', 'pc');
dlmwrite([path,'/jswzy.txt'],jswzy,'delimiter','\t','precision','%d', 'newline', 'pc');
dlmwrite([path,'/jswzz.txt'],jswzz,'delimiter','\t','precision','%d', 'newline', 'pc');

%k*E* & k*H*
dlmwrite([path,'/kx_Ey.txt'],kx_Ey,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/kx_Ez.txt'],kx_Ez,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/kx_Hy.txt'],kx_Hy,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/kx_Hz.txt'],kx_Hz,'delimiter','\t','precision',10, 'newline', 'pc');

dlmwrite([path,'/ky_Ex.txt'],ky_Ex,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/ky_Ez.txt'],ky_Ez,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/ky_Hx.txt'],ky_Hx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/ky_Hz.txt'],ky_Hz,'delimiter','\t','precision',10, 'newline', 'pc');

dlmwrite([path,'/kz_Ex.txt'],kz_Ex,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/kz_Ey.txt'],kz_Ey,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/kz_Hx.txt'],kz_Hx,'delimiter','\t','precision',10, 'newline', 'pc');
dlmwrite([path,'/kz_Hy.txt'],kz_Hy,'delimiter','\t','precision',10, 'newline', 'pc');

%source
dlmwrite([path,'/source.txt'],source,'delimiter','\t','precision',10, 'newline', 'pc');
toc;