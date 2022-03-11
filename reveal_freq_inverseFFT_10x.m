clear;

%% Load the shifted FFT spectrum 
shift_sq_fft = load('X_Our_sq600_old_shiftfft.mat');
shift_fft = shift_sq_fft.shift_sq;
sq_file = load('sq_final600_um');
sq = sq_file.sq;

%% seed 1
sd = 1;
red_freq = [189,110,50,130,190];
blue_freq = [153,133,193,146,126];
red_img = [12,108,110,17];
blue_img = [371,417,425,587];

%% seed 500
% sd = 500;
% red_freq = [4849,4951,4850,4950,4949,4750];
% blue_freq = [3944,3649,3844,3744,3650,3644,4550];
% red_img = [69,186,94,77,173];
% blue_img = [433,378,339,534,419];

%% seed 700
% sd = 700;
% red_freq = [4951,4850,4849,4950,4949,4750];
% blue_freq = [3944,3744,3844,3644,3650,3649,4550];
% red_img = [167,221,3,41,103];
% blue_img = [580,363,323,447,532];

%% images 
imgs = blue_img
% imgs = [red_img,blue_img];
% imgs = sort(imgs);

%% Red freq
sp = red_freq;
sp_c = 399 - sp;
q1 = floor(sp./20);
r1 = mod(sp,20);
row1 = q1*5+1;
col1 = r1*5+1;
qc = floor(sp_c./20);
rc = mod(sp_c,20);
row_c = qc*5+1;
col_c = rc*5+1;

%% Blue freq
sp_b = blue_freq;
sp_c_b = 399 - sp_b;
q1_b = floor(sp_b./20);
r1_b = mod(sp_b,20);
row1_b = q1_b*5+1;
col1_b = r1_b*5+1;
qc_b = floor(sp_c_b./20);
rc_b = mod(sp_c_b,20);
row_c_b = qc_b*5+1;
col_c_b = rc_b*5+1;


%% Inverse FFT Red
for num = imgs
  orig_shiftfft = shift_fft(:,:,num);
  hide_tophalf = zeros(100,100);
  hide_tophalf_b = zeros(100,100);
  for i = 1:length(sp)
      row = row1(i);
      col = col1(i);
      row2 = row_c(i);
      col2 = col_c(i);

      if row == 1
          if col == 1
            %disp(fft_f(row:row+4,col:col+4));
            hide_tophalf(row:row+4,col:col+4)= orig_shiftfft(row:row+4,col:col+4);
            hide_tophalf(1,col2+1:col2+4)= orig_shiftfft(1,col2+1:col2+4);
            hide_tophalf(row2+1:row2+4,1)= orig_shiftfft(row2+1:row2+4,1);
            hide_tophalf(row2+1:row2+4,col2+1:col2+4)= orig_shiftfft(row2+1:row2+4,col2+1:col2+4);
          end
          if col ~= 1
            %disp(fft_f(row:row+4,col:col+4));
            hide_tophalf(row:row+4,col:col+4)= orig_shiftfft(row:row+4,col:col+4);
            hide_tophalf(1,col2+1:col2+5)= orig_shiftfft(1,col2+1:col2+5);
            hide_tophalf(row2+1:row2+4,col2+1:col2+5)= orig_shiftfft(row2+1:row2+4,col2+1:col2+5);
          end
      end

      if row ~= 1
          if col == 1
            %disp(fft_f(row:row+4,col:col+4));
            hide_tophalf(row:row+4,col:col+4)= orig_shiftfft(row:row+4,col:col+4);
            hide_tophalf(row2+1:row2+5,1)= orig_shiftfft(row2+1:row2+5,1);
            hide_tophalf(row2+1:row2+5,col2+1:col2+4)= orig_shiftfft(row2+1:row2+5,col2+1:col2+4);
          end
          if col ~= 1
            %disp(fft_f(row:row+4,col:col+4));
            hide_tophalf(row:row+4,col:col+4)= orig_shiftfft(row:row+4,col:col+4);
            hide_tophalf(row2+1:row2+5,col2+1:col2+5)= orig_shiftfft(row2+1:row2+5,col2+1:col2+5);
          end
      end
  end
  %% Inverse FFT Blue
  for i = 1:length(sp_b)
      row = row1_b(i);
      col = col1_b(i);
      row2 = row_c_b(i);
      col2 = col_c_b(i);

      if row == 1
          if col == 1
            %disp(fft_f(row:row+4,col:col+4));
            hide_tophalf_b(row:row+4,col:col+4)= orig_shiftfft(row:row+4,col:col+4);
            hide_tophalf_b(1,col2+1:col2+4)= orig_shiftfft(1,col2+1:col2+4);
            hide_tophalf_b(row2+1:row2+4,1)= orig_shiftfft(row2+1:row2+4,1);
            hide_tophalf_b(row2+1:row2+4,col2+1:col2+4)= orig_shiftfft(row2+1:row2+4,col2+1:col2+4);
          end
          if col ~= 1
            %disp(fft_f(row:row+4,col:col+4));
            hide_tophalf_b(row:row+4,col:col+4)= orig_shiftfft(row:row+4,col:col+4);
            hide_tophalf_b(1,col2+1:col2+5)= orig_shiftfft(1,col2+1:col2+5);
            hide_tophalf_b(row2+1:row2+4,col2+1:col2+5)= orig_shiftfft(row2+1:row2+4,col2+1:col2+5);
          end
      end

      if row ~= 1
          if col == 1
            %disp(fft_f(row:row+4,col:col+4));
            hide_tophalf_b(row:row+4,col:col+4)= orig_shiftfft(row:row+4,col:col+4);
            hide_tophalf_b(row2+1:row2+5,1)= orig_shiftfft(row2+1:row2+5,1);
            hide_tophalf_b(row2+1:row2+5,col2+1:col2+4)= orig_shiftfft(row2+1:row2+5,col2+1:col2+4);
          end
          if col ~= 1
            %disp(fft_f(row:row+4,col:col+4));
            hide_tophalf_b(row:row+4,col:col+4)= orig_shiftfft(row:row+4,col:col+4);
            hide_tophalf_b(row2+1:row2+5,col2+1:col2+5)= orig_shiftfft(row2+1:row2+5,col2+1:col2+5);
          end
      end
  end
  inv_img_red = ifft2(ifftshift(hide_tophalf));
  inv_img_blue = ifft2(ifftshift(hide_tophalf_b));
  bottom = min(min(min(min(sq(:,:,num)))));
  top = max(max(max(max(sq(:,:,num)))));
  figure()
  set(gcf, 'Position',  [2743,175.5,1912,440]);
  s1=subplot(1,3,1);
  surf(inv_img_red(36:65,36:65));
  set(s1, 'PlotBoxAspectRatio',  [1 1.18 1]);
  view(0,90);
  h = colorbar;
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
  xticks([0 29])
  xticklabels({num2str((815/1000)*0),'24.45 \mum'})
  yticks([0 29])
  yticklabels({num2str((815/1000)*0),'24.45 \mum'})
  a = get(gca,'XTickLabel');
  set(gca,'XTickLabel',a,'fontsize',21,'fontweight','bold')
  a = get(gca,'YTickLabel');
  set(gca,'YTickLabel',a,'fontsize',21,'fontweight','bold')
  title({['IFFT Height Map'],'(\color{red}Red \color{black}top 2.5% global bands)'},'fontsize',20,'fontweight','bold');
  
  s2=subplot(1,3,2);
  surf(inv_img_blue(36:65,36:65));
  set(s2, 'PlotBoxAspectRatio', [1 1.18 1]);
  view(0,90);
  h = colorbar;
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
 xticks([0 29])
  xticklabels({num2str((815/1000)*0),'24.45 \mum'})
  yticks([0 29])
  yticklabels({num2str((815/1000)*0),'24.45 \mum'})
  a = get(gca,'XTickLabel');
  set(gca,'XTickLabel',a,'fontsize',21,'fontweight','bold')
  a = get(gca,'YTickLabel');
  set(gca,'YTickLabel',a,'fontsize',21,'fontweight','bold')
  title({['IFFT Height Map'],'(\color{blue}Blue \color{black}top 2.5% global bands)'},'fontsize',20,'fontweight','bold');
 
  s3 = subplot(1,3,3);
  surf(sq(:,:,num));
  set(s3, 'PlotBoxAspectRatio',  [1 1.18 1]);
  view(0,90);
  h = colorbar;
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
  xticks([0 29])
  xticklabels({num2str((815/1000)*0),'24.45 \mum'})
  yticks([0 29])
  yticklabels({num2str((815/1000)*0),'24.45 \mum'})
  a = get(gca,'XTickLabel');
  set(gca,'XTickLabel',a,'fontsize',21,'fontweight','bold')
  a = get(gca,'YTickLabel');
  set(gca,'YTickLabel',a,'fontsize',21,'fontweight','bold')
  title({'Original Height Map (BLUE)',''},'fontsize',22,'fontweight','bold');
%   caxis manual;
%   caxis([bottom top]);
  filename = ['ifft_800_100\',num2str(num),'.jpg'];
  saveas(gcf,filename);
  filename = ['ifft_800_100\',num2str(num),'.fig'];
  saveas(gcf,filename);
  
  %% 3D perspective
  figure()
  set(gcf, 'Position',  [2743,175.5,1912,440]);
  subplot(1,3,1);
  surf(inv_img_red(36:65,36:65));
  view(45,45);
  h = colorbar;
%   caxis manual;
%   caxis([bottom top]);
  set(h,'fontsize',18,'fontweight','bold')
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
  set(gca,'Yticklabel',[]) 
  set(gca,'Xticklabel',[])
  a = get(gca,'ZTickLabel');
  set(gca,'ZTickLabel',a,'fontsize',18,'fontweight','bold')
%   title({['IFFT Height Map'],'(\color{red}Red \color{black}top 2.5% global bands)'},'fontsize',20,'fontweight','bold');

  subplot(1,3,2);
  surf(inv_img_blue(36:65,36:65));
  view(45,45);
  h = colorbar;
%   caxis manual;
%   caxis([bottom top]);
  set(h,'fontsize',18,'fontweight','bold')
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
  set(gca,'Yticklabel',[]) 
  set(gca,'Xticklabel',[])
   a = get(gca,'ZTickLabel');
  set(gca,'ZTickLabel',a,'fontsize',18,'fontweight','bold')
%   title({['IFFT Height Map'],'(\color{blue}Blue \color{black}top 2.5% global bands)'},'fontsize',20,'fontweight','bold');
  
  subplot(1,3,3);
  surf(sq(:,:,num));
  view(45,45);
  h = colorbar;
%   caxis manual;
%   caxis([bottom top]);
  set(h,'fontsize',18,'fontweight','bold')
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
  set(gca,'Yticklabel',[]) 
  set(gca,'Xticklabel',[])
  a = get(gca,'ZTickLabel');
  set(gca,'ZTickLabel',a,'fontsize',18,'fontweight','bold')
%   title({'Original Height Map (Red)'},'fontsize',22,'fontweight','bold');

  filename = ['ifft_800_100\3D_',num2str(num),'.jpg'];
  saveas(gcf,filename);
  filename = ['ifft_800_100\3D_',num2str(num),'.fig'];
  saveas(gcf,filename);
end
