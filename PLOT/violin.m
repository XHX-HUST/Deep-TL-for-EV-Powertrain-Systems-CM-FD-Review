% 小提琴图绘制脚本
% 数据准备
X = 1:12; % 12个跨域任务

A = resultMatrix;
B = zeros(50, 12);

% 数据重塑
for i = 1:12
    % 计算当前组的起始和结束行
    start_row = (i-1)*5 + 1;
    end_row = i*5;
    
    % 提取当前组的前10列数据
    if start_row > size(A, 1) || end_row > size(A, 1)
        error('数据行数不足，无法完成分组提取');
    end
    
    group_data = A(start_row:end_row, 1:10);
    
    % 将数据重塑为列向量
    column_vector = reshape(group_data', 50, 1);
    
    % 将列向量存入结果矩阵B的第i列
    B(:, i) = column_vector;
end
Y = B;

% 设置图表样式
edgecolor = [42, 89, 145] / 255;                     % 边框颜色 (黑色)
filledcolor = [46, 114, 188] / 255;        % 填充颜色 (蓝色)
filledcolor = repmat(filledcolor, size(Y, 2), 1); % 复制颜色用于所有箱体

pos = 1:12;                                % 箱体位置

% 创建图形窗口
figure('Position', [100, 100, 460, 230]);

% 定义字体参数
font_name = 'Times New Roman';

% 绘制箱线图
box = boxplot(Y, 'positions', pos, 'Colors', edgecolor, 'Widths', 0.4, ...
              'Symbol', '+', 'OutlierSize', 2, 'Whisker', 1.5);
set(box, 'LineWidth', 0.6);

% 为箱体添加颜色填充
boxobjects = findobj(gca, 'Tag', 'Box');
for i = 1:length(boxobjects)
    patch(get(boxobjects(i), 'XData'), get(boxobjects(i), 'YData'), filledcolor(i,:), ...
          'FaceAlpha', 0.5, 'EdgeColor', edgecolor); % 设置填充颜色和透明度
end

% 设置坐标轴
set(gca, 'XTick', pos, ...
         'XTickLabel', {'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'}, ...
         'Xlim', [0 13], ...              % 扩展X轴范围
         'Ylim', [0.65 1.03], ...          % 调整Y轴范围
         'FontSize', 8.5, ...
         'FontName', font_name); 

% 添加标题和标签
xlabel('Task', 'FontName', font_name, 'FontSize', 8.5);
ylabel('Accuracy(%)', 'FontName', font_name, 'FontSize', 8.5);

% 网格和背景设置
grid on;
set(gca, 'XGrid', 'off', 'YGrid', 'on', 'GridLineStyle', '--', 'GridAlpha', 0.5); % 关闭X轴网格，保留Y轴网格
set(gcf, 'Color', 'white');

% % 添加统计信息
% stats = struct();
% for i = 1:length(pos)
%     stats(i).min = min(Y(:, i));
%     stats(i).q1 = prctile(Y(:, i), 25);
%     stats(i).median = median(Y(:, i));
%     stats(i).q3 = prctile(Y(:, i), 75);
%     stats(i).max = max(Y(:, i));
%     stats(i).mean = mean(Y(:, i));
%     stats(i).std = std(Y(:, i));
% end
% 
% % 输出统计信息到命令窗口
% fprintf('\n=== 各任务性能统计 ===\n');
% fprintf('任务\t最小值\t\tQ1\t\t中位数\t\tQ3\t\t最大值\t\t平均值\t\t标准差\n');
% for i = 1:length(pos)
%     fprintf('T%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
%         i, stats(i).min, stats(i).q1, stats(i).median, stats(i).q3, stats(i).max, stats(i).mean, stats(i).std);
% end