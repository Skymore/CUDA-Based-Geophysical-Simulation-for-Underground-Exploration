tic;
path = '/Users/taorui/Documents/GitHub/Tujian_github/Tujian_Linux/data_pianyi';
dlmwrite([path,'\CAEx.txt'],CAEx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\CAEy.txt'],CAEy,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\CAEz.txt'],CAEz,'delimiter',',','precision',10, 'newline', 'pc');

dlmwrite([path,'\CBEx.txt'],CBEx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\CBEy.txt'],CBEy,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\CBEz.txt'],CBEz,'delimiter',',','precision',10, 'newline', 'pc');

dlmwrite([path,'\CPHx.txt'],CPHx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\CPHy.txt'],CPHy,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\CPHz.txt'],CPHz,'delimiter',',','precision',10, 'newline', 'pc');

dlmwrite([path,'\CQHx.txt'],CQHx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\CQHy.txt'],CQHy,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\CQHz.txt'],CQHz,'delimiter',',','precision',10, 'newline', 'pc');

%RAE
dlmwrite([path,'\RAExy.txt'],RAExy,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RAExz.txt'],RAExz,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RAEyx.txt'],RAEyx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RAEyz.txt'],RAEyz,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RAEzx.txt'],RAEzx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RAEzy.txt'],RAEzy,'delimiter',',','precision',10, 'newline', 'pc');

%RAH
dlmwrite([path,'\RAHxy.txt'],RAHxy,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RAHxz.txt'],RAHxz,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RAHyx.txt'],RAHyx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RAHyz.txt'],RAHyz,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RAHzx.txt'],RAHzx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RAHzy.txt'],RAHzy,'delimiter',',','precision',10, 'newline', 'pc');

%RBE
dlmwrite([path,'\RBExy.txt'],RBExy,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RBExz.txt'],RBExz,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RBEyx.txt'],RBEyx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RBEyz.txt'],RBEyz,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RBEzx.txt'],RBEzx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RBEzy.txt'],RBEzy,'delimiter',',','precision',10, 'newline', 'pc');

%RBH
dlmwrite([path,'\RBHxy.txt'],RBHxy,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RBHxz.txt'],RBHxz,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RBHyx.txt'],RBHyx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RBHyz.txt'],RBHyz,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RBHzx.txt'],RBHzx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\RBHzy.txt'],RBHzy,'delimiter',',','precision',10, 'newline', 'pc');

%fsw
dlmwrite([path,'\fswzx.txt'],fswzx,'delimiter',',','precision','%d', 'newline', 'pc');
dlmwrite([path,'\fswzy.txt'],fswzy,'delimiter',',','precision','%d', 'newline', 'pc');
dlmwrite([path,'\fswzz.txt'],fswzz,'delimiter',',','precision','%d', 'newline', 'pc');

dlmwrite([path,'\jswzx.txt'],jswzx,'delimiter',',','precision','%d', 'newline', 'pc');
dlmwrite([path,'\jswzy.txt'],jswzy,'delimiter',',','precision','%d', 'newline', 'pc');
dlmwrite([path,'\jswzz.txt'],jswzz,'delimiter',',','precision','%d', 'newline', 'pc');

%k*E* & k*H*
dlmwrite([path,'\kx_Ey.txt'],kx_Ey,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\kx_Ez.txt'],kx_Ez,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\kx_Hy.txt'],kx_Hy,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\kx_Hz.txt'],kx_Hz,'delimiter',',','precision',10, 'newline', 'pc');

dlmwrite([path,'\ky_Ex.txt'],ky_Ex,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\ky_Ez.txt'],ky_Ez,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\ky_Hx.txt'],ky_Hx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\ky_Hz.txt'],ky_Hz,'delimiter',',','precision',10, 'newline', 'pc');

dlmwrite([path,'\kz_Ex.txt'],kz_Ex,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\kz_Ey.txt'],kz_Ey,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\kz_Hx.txt'],kz_Hx,'delimiter',',','precision',10, 'newline', 'pc');
dlmwrite([path,'\kz_Hy.txt'],kz_Hy,'delimiter',',','precision',10, 'newline', 'pc');

%source
dlmwrite([path,'\source.txt'],source,'delimiter',',','precision',10, 'newline', 'pc');
toc;