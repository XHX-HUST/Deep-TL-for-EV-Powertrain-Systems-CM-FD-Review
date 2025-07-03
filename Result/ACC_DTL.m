clc
clear all

% 定义根路径
outPath = 'E:\Project_py\Review\DTL\Result\sv\';

num = 1;

if num == 1
    rootPath = 'E:\Project_py\Review\DTL\checkpoint\Base_0530-171924\';
elseif num == 2
    rootPath = 'E:\Project_py\Review\DTL\checkpoint\PTFT_0531-211549\';
elseif num == 3
    rootPath = 'E:\Project_py\Review\DTL\checkpoint\SMM_0530-223104\';
elseif num == 4
    rootPath = 'E:\Project_py\Review\DTL\checkpoint\DAT_0531-090634\';
end

% 获取根路径下的所有子文件夹（如01、02等）
subfolders = dir(fullfile(rootPath, '*'));
subfolders = {subfolders([subfolders.isdir]).name}; % 筛选文件夹名称
subfolders(ismember(subfolders, {'.', '..'})) = []; % 移除.和..

% 初始化结果矩阵 (5*12行 x 10+1列)
resultMatrix = zeros(5*length(subfolders), 11);
resultMatrix1 = zeros(length(subfolders), 8);

% 遍历子文件夹
for folderIdx = 1:length(subfolders)
    folderPath = fullfile(rootPath, subfolders{folderIdx});
    disp(folderPath)
    matFiles = dir(fullfile(folderPath, '*.mat')); % 获取该子文件夹下的所有.mat文件
    
    % 遍历每个.mat文件（共5个）
    for fileIdx = 1:5 % 或 length(matFiles)，确保每个子文件夹有5个文件
        matFilePath = fullfile(folderPath, matFiles(fileIdx).name);
        data = load(matFilePath, 'ACC_sv'); % 加载ACC_st变量

        % 提取Acc_st的最后10个元素并存储到矩阵
        rowIdx = (folderIdx-1)*5 + fileIdx;  % 计算当前行索引
        resultMatrix(rowIdx, 1:10) = data.ACC_sv(end-9:end);  % 提取最后10个元素
        resultMatrix(rowIdx, 11) = mean(resultMatrix(rowIdx, 1:10)); % 平均值
    end

    % 5-MAX-MIN-AVE
    resultMatrix1(folderIdx,1:5) = resultMatrix((folderIdx*5-4):(folderIdx*5),end)';
    resultMatrix1(folderIdx,6) = max(resultMatrix((folderIdx*5-4):(folderIdx*5),end));
    resultMatrix1(folderIdx,7) = min(resultMatrix((folderIdx*5-4):(folderIdx*5),end));
    resultMatrix1(folderIdx,8) = mean(resultMatrix((folderIdx*5-4):(folderIdx*5),end));
end

% 保存结果矩阵到MAT文件
if num == 1
    save(fullfile(outPath, 'Base_ACC_sv_per1.mat'), 'resultMatrix');
    save(fullfile(outPath, 'Base_ACC_sv_per2.mat'), 'resultMatrix1');
elseif num == 2
    save(fullfile(outPath, 'PTFT_ACC_sv_per1.mat'), 'resultMatrix');
    save(fullfile(outPath, 'PTFT_ACC_sv_per2.mat'), 'resultMatrix1');
elseif num == 3
    save(fullfile(outPath, 'SMM_ACC_sv_per1.mat'), 'resultMatrix');
    save(fullfile(outPath, 'SMM_ACC_sv_per2.mat'), 'resultMatrix1');
elseif num == 4
    save(fullfile(outPath, 'DAT_ACC_sv_per1.mat'), 'resultMatrix');
    save(fullfile(outPath, 'DAT_ACC_sv_per2.mat'), 'resultMatrix1');
end

