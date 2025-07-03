% 示例数据（替换为您的真实标签和预测标签）
trueLabels = Labels.matrix(1,:);       % 真实标签（1-3类）
predLabels = Labels.matrix(2,:);  % 预测标签（1-3类）

% 自定义颜色映射 - 这里使用蓝到红的渐变
% 你可以修改此部分来自定义颜色
colorMap = [ones(10001,1), linspace(0,1,10001)', zeros(10001,1)];

% 创建混淆矩阵
classes = unique([trueLabels; predLabels]);
numClasses = length(classes);
confMatrix = zeros(numClasses);

for i = 1:length(trueLabels)
    trueIdx = find(classes == trueLabels(i));
    predIdx = find(classes == predLabels(i));
    confMatrix(trueIdx, predIdx) = confMatrix(trueIdx, predIdx) + 1;
end

% 归一化混淆矩阵以便于颜色映射
normConfMatrix = round(confMatrix ./ sum(confMatrix, 2),4);

% 创建三维柱形图 - 设置背景透明
figure('Position', [100, 100, 500, 500]); % 设置窗口大小
hold on;

% 设置柱形参数
barWidth = 0.8;    % 柱体在x轴方向的宽度(预测标签方向)
barDepth = 0.8;    % 柱体在y轴方向的深度(真实标签方向)

% 绘制每个柱体
for i = 1:numClasses
    for j = 1:numClasses
        % 计算柱体位置
        x = j;
        y = i;
        z = 0;

        % 获取柱体高度和颜色
        height = normConfMatrix(i, j);
        colorIdx = normConfMatrix(i, j)*10000 + 1;
        barColor = colorMap(colorIdx,:);

        if height < 0.001
            height = 0.001;
        end

        % 创建一个长方体表示柱体
        % 定义长方体的顶点
        vertices = [
            x-barWidth/2, y-barDepth/2, z;
            x+barWidth/2, y-barDepth/2, z;
            x+barWidth/2, y+barDepth/2, z;
            x-barWidth/2, y+barDepth/2, z;
            x-barWidth/2, y-barDepth/2, z+height;
            x+barWidth/2, y-barDepth/2, z+height;
            x+barWidth/2, y+barDepth/2, z+height;
            x-barWidth/2, y+barDepth/2, z+height
            ];

        % 定义长方体的面
        faces = [
            1, 2, 3, 4;  % 底面
            5, 6, 7, 8;  % 顶面
            1, 2, 6, 5;  % 前面
            2, 3, 7, 6;  % 右面
            3, 4, 8, 7;  % 后面
            4, 1, 5, 8   % 左面
            ];

        % 绘制长方体
        patch('Faces', faces, 'Vertices', vertices, ...
            'FaceColor', barColor, 'EdgeAlpha', 0.3, ...
            'FaceAlpha', 0.9);

    end
end

% 设置坐标轴范围和刻度
xlim([0.4, numClasses+0.6]);
ylim([0.4, numClasses+0.6]);
zlim([0, 1.1]);

% 设置刻度标签
xticks(1:numClasses);
xticklabels(classes);
yticks(1:numClasses);
yticklabels(classes);

% 设置刻度值字体大小
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

% 美化图形
grid on;
box on;
view(45,40);


% 指定保存路径和文件名
savePath = fullfile('E:\Project_py\Review\DTL\Result\Confusion\', 'confusion_matrix_DAT.png'); 
% 检查目录是否存在，如果不存在则创建
[saveDir, ~, ~] = fileparts(savePath);
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% 保存图像到指定位置，设置分辨率和大小
print(gcf, '-dpng', '-r600', savePath);