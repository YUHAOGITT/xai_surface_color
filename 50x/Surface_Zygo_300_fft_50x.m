clear;

%% Load datasets
filename = 'Zygo_mat\10x_';
num = [13,14,15,19,20,21];

j = 1;
suffix = '.mat';
for i = num
     load_sur=load(strcat(filename,int2str(i)));
     surface(:,:,j)=load_sur.s;
     j = j+1;
end

% unit: m to um
surface_1 = double(surface(:,:,1))*1.88*10^(-3);
surface_2 = double(surface(:,:,2))*1.88*10^(-3);
surface_3 = double(surface(:,:,3))*1.88*10^(-3);
surface_4 = double(surface(:,:,4))*1.88*10^(-3);
surface_5 = double(surface(:,:,5))*1.88*10^(-3);
surface_6 = double(surface(:,:,6))*1.88*10^(-3);

red_50x_part(:,:,1) = surface_4;
red_50x_part(:,:,2) = surface_5;
red_50x_part(:,:,3) = surface_6;
blue_50x_part(:,:,1) = surface_1;
blue_50x_part(:,:,2) = surface_2;
blue_50x_part(:,:,3) = surface_3;

square_size = 200;

overlap = 0;
w_size = fix((1000-square_size)/(square_size*(1-overlap)))+1;
l_size = fix((1000-square_size)/(square_size*(1-overlap)))+1;

%% 50x
num_s =3;
red_part = red_50x_part;
blue_part = blue_50x_part;

%% divide maps into 100x100 squares
for k = 1:num_s
    lhs_l(k,:) = [0:l_size];
    lhs_w(k,:) = [0:w_size];
end
s_length = lhs_l*(square_size*(1-overlap));
s_width = lhs_w*(square_size*(1-overlap));

%% Divide height maps
z=1;
for k =1:num_s
  for i = 1:l_size
     for j = 1:w_size
     transition_square(:,:,z) = red_part(s_length(k,i)+1:s_length(k,i)+square_size,s_width(k,j)+1:s_width(k,j)+square_size,k);
     z = z+1;
     end;
  end;
end;

z=1;
for k =1:num_s
  for i = 1:l_size
     for j = 1:w_size
     blue_square(:,:,z) = blue_part(s_length(k,i)+1:s_length(k,i)+square_size,s_width(k,j)+1:s_width(k,j)+square_size,k);
     z = z+1;
     end;
  end;
end;

%% zero-pad
transition_square_pad = padarray(transition_square,[150 150],0,'both');
blue_square_pad = padarray(blue_square,[150 150],0,'both');
% w = blackmanharris(100).*blackmanharris(100)';

%% FFT
for i = 1:(l_size*w_size*num_s)
  f_t(:,:,i) = fft2(transition_square_pad(:,:,i));
  f_b(:,:,i) = fft2(blue_square_pad(:,:,i));
end;

% a = fft2(transition_square(:,:,1));

for i = 1:(l_size*w_size*num_s)
  shift_f_t(:,:,i) = fftshift(f_t(:,:,i));
  absf_t(:,:,i) = abs(shift_f_t(:,:,i));
  shift_f_b(:,:,i) = fftshift(f_b(:,:,i));
  absf_b(:,:,i) = abs(shift_f_b(:,:,i));
end

shift_fft = cat(3,shift_f_t(:,:,1:l_size*w_size*num_s),shift_f_b(:,:,1:l_size*w_size*num_s));
X = cat(3,absf_t(:,:,1:l_size*w_size*num_s),absf_b(:,:,1:l_size*w_size*num_s));
%X(51,51,:) = 0;
X_tophalf = cat(3,absf_t(1:251,:,1:l_size*w_size*num_s),absf_b(1:251,:,1:l_size*w_size*num_s));
%X_tophalf(51,51,:) = 0;
Y = cat(1,zeros(l_size*w_size*num_s,1),ones(l_size*w_size*num_s,1));
S =  cat(3,transition_square(:,:,1:l_size*w_size*num_s),blue_square(:,:,1:l_size*w_size*num_s));
