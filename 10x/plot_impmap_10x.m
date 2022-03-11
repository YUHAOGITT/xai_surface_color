clear;
region = 'T';
sd = '300';
map = load(strcat('sq600_10x_our_',region,'_posi_impmap_sd',sd,'.mat'));
imp_map = map.imp_map;
map2 = max(max(imp_map))*0.01*(imp_map.^2);

figure();
set(gcf, 'Position',  [1, 1, 1550, 810]);
imagesc(map2);
colormap(gray);
xticks([1 50 100])
xticklabels({'-6.13','0','6.13'})
yticks([1.5 49.5])
yticklabels({'6.13','0'})
a = get(gca,'XTickLabel');  
b = get(gca,'YTickLabel');  
set(gca,'XTickLabel',a,'fontsize',31,'FontWeight','bold');
set(gca,'YTickLabel',b,'fontsize',31,'FontWeight','bold');
ylabel('Frequency (10^{-4} 1/nm)','fontsize',36,'FontWeight','bold');
xlabel('Frequency (10^{-4} 1/nm)','fontsize',36,'FontWeight','bold');
title('Global positive importance towards \color{red}Red','fontsize',38,'FontWeight','bold');

q1 = min(map2(:));
q2 = (max(map2(:))+min(map2(:)))/2;
q3 = max(map2(:))*0.9;
h = colorbar;
ylabel(h,{'Global importance'},'fontsize',36,'fontweight','bold');
set(h, 'YTick', [q1 q2 q3]);
Y_str = string([round(sqrt(q1/(max(max(imp_map))*0.01)),0) round(sqrt(q2/(max(max(imp_map))*0.01)),0) round(sqrt(q3/(max(max(imp_map))*0.01)),0)]);
set(h, 'YTickLabel', Y_str);

% filename = ['lime_zygo_wrong\imp_map\zygo_correct_model_',region,'_nega_map_sd',sd,'.jpg'];
% saveas(gcf,filename);
filename = ['imp_map\sq600_10x_our_',region,'_posi_map_sd',sd,'.png'];
saveas(gcf,filename);
