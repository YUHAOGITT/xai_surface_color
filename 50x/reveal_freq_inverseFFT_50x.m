clear;

%% Load the shifted FFT spectrum 
shift_sq_fft = load('X_sq_500_shift.mat');
shift_fft = shift_sq_fft.shift_sq;
sq_file = load('sq_50x_600.mat');
sq = sq_file.sq_50x;

%% seed 1
% sd = 1;
% red_freq = [4951,4850,4950,4849,4949,4750];
% blue_freq = [3744,3844,3944,3649,3644,3650];
red_img = [191,4,226,18,24];
% blue_img = [405,586,449,529,314];
% % % blue_img = [310,401,447,468,474];

%% seed 500
% sd = 500;
% red_freq = [4849,4951,4850,4950,4949,4750];
% blue_freq = [3944,3649,3844,3744,3650,3644,4550];
% red_img = [69,186,94,77,173];
% blue_img = [433,378,339,534,419];
% % % % blue_img = [401,445,447,468,474];

%% seed 700
sd = 700;
red_freq = [4950, 4849, 4848, 4949, 4948, 4750, 4947, 4749, 4850, 4449, 4847, 4651, 4751, 4650, 4649, 4851, 850, 1149, 4448, 4748, 849, 1150, 449, 749, 4450, 550, 4551, 4846, 1050, 750, 4648, 4860, 1002, 2819, 3236, 3907, 4051, 1152, 737, 2083, 313, 216, 3022, 742, 1046, 3673, 852, 4951, 1701, 4550, 2436, 751, 4834, 1478, 1949, 4461, 743, 851, 4254, 926, 1916, 1989, 2969, 4555, 1449, 899, 3601, 1049, 4936, 1251, 911, 4349, 1068, 451, 1774, 1055, 2330, 1229, 2070, 4370, 528, 3180, 321, 4229, 837, 4374, 579, 2712, 853, 4071, 618, 49, 1120, 3709, 4249, 1099, 1413, 3351, 2663, 2317, 4048, 3160, 109, 1092, 3154, 3366, 1239, 1261, 4806, 2569, 4142, 4688, 1335, 2032, 794, 1172, 2848, 1151, 4332, 4916, 4930, 1673, 2916, 1040, 2175];
blue_freq = [3944, 3844, 3649, 3744, 3650, 3644, 4550, 3957, 3958, 3743, 3855, 4549, 4457, 3645, 4458, 4755, 3749, 3658, 3643, 4558, 3544, 4456, 3843, 3956, 3945, 2944, 3838, 4655, 4044, 4544, 4954, 4547, 3955, 2844, 3845, 4551, 4756, 3857, 3758, 3745, 3044, 3856, 3750, 3656, 3943, 4451, 3642, 3757, 2843, 3854, 3754, 3657, 3651, 3755, 4644, 3652, 3756, 3664, 3558, 3738, 3748, 3950, 2855, 3858, 3641, 4058, 3747, 3851, 4754, 4645, 4444, 3938, 4055, 2858, 4054, 3449, 3058, 2943, 4452, 3661, 4057, 3839, 4844, 3751, 4554, 3739, 3761, 2958, 4038, 3648, 4557, 3741, 4545, 3045, 4662, 3144, 3964, 2838, 3742, 2942, 4459, 3647, 3444, 4814, 3556, 4358, 3837, 4045, 4758, 3864, 2758, 4656, 4356, 4665, 3157, 1564, 3557, 4352, 3541, 4445, 3737, 3543, 4543, 3549, 4955];
% red_img = [3,41,103,167,221];
blue_img = [580,363,323,447,532]; %
% % blue_img = [310,401,447,468,474];

%% images 
imgs = red_img;
% imgs = sort(imgs);

%% Red freq
sp = red_freq;
sp_c = 9999 - sp;
q1 = floor(sp./100);
r1 = mod(sp,100);
row1 = q1*5+1;
col1 = r1*5+1;
qc = floor(sp_c./100);
rc = mod(sp_c,100);
row_c = qc*5+1;
col_c = rc*5+1;

%% Blue freq
sp_b = blue_freq;
sp_c_b = 9999 - sp_b;
q1_b = floor(sp_b./100);
r1_b = mod(sp_b,100);
row1_b = q1_b*5+1;
col1_b = r1_b*5+1;
qc_b = floor(sp_c_b./100);
rc_b = mod(sp_c_b,100);
row_c_b = qc_b*5+1;
col_c_b = rc_b*5+1;


%% Inverse FFT Red
for num = imgs
  orig_shiftfft = shift_fft(:,:,num);
  hide_tophalf = zeros(500,500);
  hide_tophalf_b = zeros(500,500);
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
  surf(inv_img_red(236:265,236:265));
  set(s1, 'PlotBoxAspectRatio',  [1 1.18 1]);
  view(0,90);
  h = colorbar;
%   caxis manual;
%   caxis([bottom top]);
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
  xticks([0 15 29])
  xticklabels({num2str((174/1000)*0),num2str((174/1000)*15),'5.22 \mum'})
  yticks([0 15 29])
  yticklabels({num2str((174/1000)*0),num2str((174/1000)*15),'5.22 \mum'})
  a = get(gca,'XTickLabel');
  set(gca,'XTickLabel',a,'fontsize',21,'fontweight','bold')
  a = get(gca,'YTickLabel');
  set(gca,'YTickLabel',a,'fontsize',21,'fontweight','bold')
  title({['IFFT Height Map'],'(\color{red}Red \color{black}top 2.5% global bands)'},'fontsize',20,'fontweight','bold');
  
  s2=subplot(1,3,2);
  surf(inv_img_blue(236:265,236:265));
  set(s2, 'PlotBoxAspectRatio', [1 1.18 1]);
  view(0,90);
  h = colorbar;
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
  xticks([0 15 29])
  xticklabels({num2str((174/1000)*0),num2str((174/1000)*15),'5.22 \mum'})
  yticks([0 15 29])
  yticklabels({num2str((174/1000)*0),num2str((174/1000)*15),'5.22 \mum'})
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
  xticks([0 15 29])
  xticklabels({num2str((174/1000)*0),num2str((174/1000)*15),'5.22 \mum'})
  yticks([0 15 29])
  yticklabels({num2str((174/1000)*0),num2str((174/1000)*15),'5.22 \mum'})
  a = get(gca,'XTickLabel');
  set(gca,'XTickLabel',a,'fontsize',21,'fontweight','bold')
  a = get(gca,'YTickLabel');
  set(gca,'YTickLabel',a,'fontsize',21,'fontweight','bold')
  title({'Original Height Map (RED)',''},'fontsize',22,'fontweight','bold');
%   caxis manual;
%   caxis([bottom top]);
  filename = ['Matlab_result3\ifft_300\sd',num2str(sd),'\',num2str(num),'.jpg'];
  saveas(gcf,filename);
  filename = ['Matlab_result3\ifft_300\sd',num2str(sd),'\',num2str(num),'.fig'];
  saveas(gcf,filename);
  
  %% 3D perspective
  figure()
  set(gcf, 'Position',  [2743,175.5,1912,440]);
  subplot(1,3,1);
  surf(inv_img_red(236:265,236:265));
  view(45,45);
  h = colorbar;
%   caxis manual;
%   caxis([bottom top]);
  set(h,'fontweight','bold')
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
  set(gca,'Yticklabel',[]) 
  set(gca,'Xticklabel',[])
  a = get(gca,'ZTickLabel');
  set(gca,'ZTickLabel',a,'fontsize',18,'fontweight','bold')
%   title({['IFFT Height Map ',num2str(num)],'(\color{red}Red \color{black}frequency signatures)'},'fontsize',22,'fontweight','bold');

  subplot(1,3,2);
  surf(inv_img_blue(236:265,236:265));
  view(45,45);
  h = colorbar;
%   caxis manual;
%   caxis([bottom top]);
  set(h,'fontweight','bold')
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
  set(gca,'Yticklabel',[]) 
  set(gca,'Xticklabel',[])
   a = get(gca,'ZTickLabel');
  set(gca,'ZTickLabel',a,'fontsize',18,'fontweight','bold')
%   title({['IFFT Height Map ',num2str(num)],'(\color{blue}Blue \color{black}frequency signatures)'},'fontsize',22,'fontweight','bold');
  
  subplot(1,3,3);
  surf(sq(:,:,num));
  view(45,45);
  h = colorbar;
%   caxis manual;
%   caxis([bottom top]);
  set(h,'fontweight','bold')
  ylabel(h,'(\mum)','fontsize',19,'fontweight','bold');
  set(gca,'Yticklabel',[]) 
  set(gca,'Xticklabel',[])
  a = get(gca,'ZTickLabel');
  set(gca,'ZTickLabel',a,'fontsize',18,'fontweight','bold')
%   title({'Original Height Map ',num2str(num)},'fontsize',22,'fontweight','bold');

  filename = ['Matlab_result3\ifft_300\sd',num2str(sd),'\3D_',num2str(num),'.jpg'];
  saveas(gcf,filename);
  filename = ['Matlab_result3\ifft_300\sd',num2str(sd),'\3D_',num2str(num),'.fig'];
  saveas(gcf,filename);
end
