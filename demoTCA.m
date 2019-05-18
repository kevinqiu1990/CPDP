% init program
clc();
clear();
addpath (genpath('.'))

% super-parameter
sigma=1;
dim=10;
mu=1;
lambda=1;

% set result file
learnerName = 'LR';
modelName = 'TCA';
file_name=['./output/',modelName,'_',learnerName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,dim,mu,lambda,dataset,target,source,f1,AUC';
fprintf(file,'%s\n',headerStr);

% Select dataset
for dataset = [1,2]
    if dataset == 1
        dataName = 'AEEEM';
        load ./data/AEEEM.mat
        fileList={'EQ','JDT'};
        attributeNum=61;
        labelIndex=62;
    elseif dataset == 2
        dataName = 'Promise';
        load ./data/promise.mat
        fileList={'ant','arc'};
        attributeNum=20;
        labelIndex=21;
    end
    
    % Select target project
    for i = 1:length(fileList)
        targetName=fileList{i};
        targetData=eval(targetName);
        targetData(targetData(:,labelIndex)==-1,labelIndex)=0;
        targetX = targetData(:,1:attributeNum);
        targetX = zscore(targetX);
        targetY = targetData(:,labelIndex);
        
        % Select source project
        for j = 1:length(fileList)
            sourceName=fileList{j};
            if(i~=j)
                sourceData=eval(sourceName);
                sourceData(sourceData(:,labelIndex)==-1,labelIndex)=0;
                sourceX = sourceData(:,1:attributeNum);
                sourceX = zscore(sourceX);
                sourceY = sourceData(:,labelIndex);
                
                % call TCA
                options = tca_options('Kernel', 'linear', 'KernelParam', sigma, 'Mu', mu, 'lambda', lambda, 'Dim', dim);
                [newtrainX, ~, newtestX] = tca(sourceX, targetX, targetX, options);
                
                % Logistic regression
                model = train([], sourceY, sparse(newtrainX), '-s 0 -c 1');
                predictY = predict(targetY, sparse(newtestX), model);
                [accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(predictY, targetY);
                
                %parameter string
                resultStr = [modelName,',',learnerName,',',num2str(dim),',',num2str(mu),',',num2str(lambda),',',dataName,',',targetName,',',sourceName,',',num2str(f_measure),',',num2str(AUC)]
                fprintf(file,'%s\n',resultStr);
            end
        end
    end
end