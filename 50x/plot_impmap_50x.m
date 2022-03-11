clear;
region = 'B';
sd = '500';
num_s = '300';
map = load(strcat('sq600_50x_',region,'_posi_map_sd',sd,'_',num_s,'.mat'));
imp_map = map.imp_map;
map2 = imp_map.^2;

figure();
set(gcf, 'Position',  [1, 1, 1550, 810]);
imagesc(map2);
colormap(gray);
xticks([1 250 500])
xticklabels({'-28.7','0','28.7'})
yticks([3 245])
yticklabels({'28.7','0'})
a = get(gca,'XTickLabel');  
b = get(gca,'YTickLabel');  
set(gca,'XTickLabel',a,'fontsize',31,'FontWeight','bold');
set(gca,'YTickLabel',b,'fontsize',31,'FontWeight','bold');
ylabel('Frequency (10^{-4} 1/nm)','fontsize',36,'FontWeight','bold');
xlabel('Frequency (10^{-4} 1/nm)','fontsize',36,'FontWeight','bold');
title('Global positive importance towards \color{blue}Blue','fontsize',38,'FontWeight','bold');

q1 = min(map2(:));
q2 = (max(map2(:))+min(map2(:)))/2;
q3 = max(map2(:))*0.9;
h = colorbar;
ylabel(h,{'Global importance'},'fontsize',36,'fontweight','bold');
set(h, 'YTick', [q1 q2 q3]);
Y_str = string([round(sqrt(q1),0) round(sqrt(q2),0) round(sqrt(q3),0)]);
set(h, 'YTickLabel', Y_str);

filename = ['imp_map\sq600_50x_',region,'_posi_map_sd',sd,'_',num_s,'.jpg'];
saveas(gcf,filename);
filename = ['imp_map\sq600_50x_',region,'_posi_map_sd',sd,'_',num_s,'.fig'];
saveas(gcf,filename);
