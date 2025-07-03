data = [0.999427083333333, 0.999427083333333, 0.999427083333333, 1, 0.999739583333333, 0.999739583333333, 1, 1, 1, 1, 1, 1; 
    0.970703125000000, 0.939869791666667, 0.834114583333333, 0.937838541666667, 0.996276041666667, 0.830963541666667, 0.937421875000000, 0.908255208333333, 0.901692708333333, 0.764114583333334, 0.731276041666667, 0.837708333333333];

figure('Color', 'white', 'Position', [100, 100, 460, 300]);

font_name = 'Times New Roman';

% 绘制分组柱状图
hBar = bar(data', 'EdgeColor', 'black', 'BarWidth', 0.8);
set(gca, 'ColorOrder', [51 111 189; 204 85 28]./255, 'NextPlot', 'replacechildren');

% 获取柱子宽度和柱子之间的间距
bar_width = hBar(1).BarWidth * 0.36;
n_methods = size(data, 1);
group_width = bar_width * n_methods;

% 添加顶部黑色圆点
hold on;
for method_idx = 1:n_methods
    for task_idx = 1:size(data, 2)
        % 计算当前柱子的X中心位置（考虑分组偏移）
        group_center = task_idx; % 组中心位置
        method_offset = (method_idx - (n_methods+1)/2) * bar_width; % 方法偏移量
        x = group_center + method_offset; % 柱子实际中心
        
        % 获取柱子高度
        y = hBar(method_idx).YData(task_idx);
        
        % 绘制黑色填充圆点
        plot(x, y, 'ko', 'MarkerSize', 1.5, 'MarkerFaceColor', 'k');
    end
end
hold off;

% 坐标轴设置
xlabel('Task', 'FontName', font_name, 'FontSize', 8.5);
ylabel('Accuracy(%)', 'FontName', font_name, 'FontSize', 8.5);

% 设置Y轴范围
ylim([0.7, 1.01]);

xticks(1:12);
xticklabels(arrayfun(@(x) sprintf('T%d', x), 1:12, 'UniformOutput', false));
set(gca, 'FontName', font_name, 'FontSize', 8.5);

% 图例
legend('S test', 'T test', 'Location', 'northeast');