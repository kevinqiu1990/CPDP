% init program
clc();
clear();
addpath (genpath('.'))

% set result file
learnerName = 'LR';
modelName = 'DG';
file_name=['./output/',modelName,'_',learnerName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,dataset,target,source,f1,AUC';
fprintf(file,'%s\n',headerStr);

% Select dataset
for dataset = [1,2]
    if dataset == 1
        dataName = 'AEEEM';
        load ./data/AEEEM.mat
        fileList={'EQ','JDT','LC','ML', 'PDE'};
        attributeNum=61;
        labelIndex=62;
    elseif dataset == 2
        dataName = 'Promise';
        load ./data/promise.mat
        fileList={'ant','arc','camel','elearning','ivy','prop','synapse','systemdata','tomcat','velocity'};
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
                
                % call DG
                alpha_weight = cal_data_gravitation(targetX, sourceX);
%                 alpha_weight = NormalizeAlpha(alpha_weight, 1);
                
                % Logistic regression
                model = train(alpha_weight, sourceY, sparse(sourceX), '-s 0 -c 1');
                predictY = predict(targetY, sparse(targetX), model);
                [accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(predictY, targetY);
                
                %parameter string
                resultStr = [modelName,',',learnerName,',',dataName,',',targetName,',',sourceName,',',num2str(f_measure),',',num2str(AUC)]
                fprintf(file,'%s\n',resultStr);
            end
        end
    end
end