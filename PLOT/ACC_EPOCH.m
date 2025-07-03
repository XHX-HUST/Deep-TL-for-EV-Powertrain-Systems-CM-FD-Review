clc
clear all

%% 数据准备
epochs = 12;
resultMatrix = zeros(4, epochs);

matFilePath1 = 'E:\Project_py\Review\DTL\Result\tv\Base_ACC_tv_per2.mat';
data1 = load(matFilePath1, 'resultMatrix1');
matFilePath2 = 'E:\Project_py\Review\DTL\Result\tv\PTFT_ACC_tv_per2.mat';
data2 = load(matFilePath2, 'resultMatrix1');
matFilePath3 = 'E:\Project_py\Review\DTL\Result\tv\SMM_ACC_tv_per2.mat';
data3 = load(matFilePath3, 'resultMatrix1');
matFilePath4 = 'E:\Project_py\Review\DTL\Result\tv\DAT_ACC_tv_per2.mat';
data4 = load(matFilePath4, 'resultMatrix1');

resultMatrix(1,:) = 100*data1.resultMatrix1(:,6)';

% 调整数据大小，便于画图  （70,95）--（90,95）
% for i=1:12
%     if resultMatrix(1,i) <= 95
%         resultMatrix(1,i) = resultMatrix(1,i)/5 + 76;
%     end
% end

resultMatrix(2,:) = 100*data2.resultMatrix1(:,6)';
resultMatrix(3,:) = 100*data3.resultMatrix1(:,6)';
resultMatrix(4,:) = 100*data4.resultMatrix1(:,6)';

resultMatrix1 = round(resultMatrix, 2);



%% 绘图设置
figure('Color', 'white', 'Position', [100, 100, 460, 230]);

% 定义字体参数
font_name = 'Times New Roman';

% 绘制曲线
% plot(1:epochs, resultMatrix(1,:), '^--r', 'MarkerSize',4, 'MarkerFaceColor','r', 'LineWidth',1.0, 'DisplayName','Base');
% hold on;
% plot(1:epochs, resultMatrix(2,:), 'b--pentagram', 'MarkerSize',5.5, 'MarkerFaceColor','b', 'LineWidth',1.0, 'DisplayName','PTFT');
% plot(1:epochs, resultMatrix(3,:), 's--', 'Color',[1 0.647 0], 'MarkerSize',5, 'MarkerFaceColor', [1 0.647 0], 'LineWidth',1.0, 'DisplayName','SMM');
% plot(1:epochs, resultMatrix(4,:), 'g--o', 'MarkerSize',4.5, 'MarkerFaceColor','g', 'LineWidth',1.0, 'DisplayName','DAT');


plot(1:epochs, resultMatrix(2,:), 'b--pentagram', 'MarkerSize',5.5, 'MarkerFaceColor','b', 'LineWidth',1.0, 'DisplayName','PTFT');
hold on;
plot(1:epochs, resultMatrix(3,:), 's--', 'Color',[1 0.647 0], 'MarkerSize',5, 'MarkerFaceColor', [1 0.647 0], 'LineWidth',1.0, 'DisplayName','SMM');
plot(1:epochs, resultMatrix(4,:), 'g--o', 'MarkerSize',4.5, 'MarkerFaceColor','g', 'LineWidth',1.0, 'DisplayName','DAT');

% 设置坐标轴标签字体
xlabel('Task', 'FontName', font_name, 'FontSize', 8.5);
ylabel('Accuracy(%)', 'FontName', font_name, 'FontSize', 8.5);

% 设置Y轴范围（关键修改点）
ylim([96.5, 100.2]);
xlim([0.7, 12.3]);

% 设置X轴刻度标签为T1,T2,T3...形式
xticklabels(arrayfun(@(x) sprintf('T%d', x), 1:epochs, 'UniformOutput', false));

% ===== 自定义Y轴刻度标记 =====
custom_yticks = [70/5+76, 75/5+76, 80/5+76, 85/5+76, 90/5+76, 95, 96, 97, 98, 99, 100];  % 自定义刻度位置
custom_yticklabels = {'70', '75', '80', '85', '90', '95', '96', '97', '98', '99', '100'};  % 自定义刻度标签

yticks(custom_yticks);
yticklabels(custom_yticklabels);

% 设置坐标轴刻度标签字体
set(gca, 'FontName', font_name, 'FontSize', 8.5, 'XTick', 1:1:epochs, 'YTick', 90:1:100);

% 设置网格线
grid on;

% 添加图例
legend('Location', 'best', 'FontName', font_name, 'FontSize', 8.5);






    