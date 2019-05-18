% init program
clc();
clear();
addpath (genpath('.'))

% super-parameter
sigma=1;
dim=10;
lambda=1;

% set result file
learnerName = 'LR';
modelName = 'JDA';
file_name=['./output/',modelName,'_',learnerName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,dim,lambda,dataset,target,source,f1,AUC';
fprintf(file,'%s\n',headerStr);

% JDA process file
process_f1_file_name = ['./output/JDA_process_f1_result.csv'];
process_f1_file=fopen(process_f1_file_name,'w');
process_AUC_file_name = ['./output/JDA_process_AUC_result.csv'];
process_AUC_file=fopen(process_AUC_file_name,'w');

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
                
                % call JDA
                JDA_options.k = dim;
                JDA_options.lambda = lambda;
                JDA_options.ker = 'linear';            % 'primal' | 'linear' | 'rbf'
                JDA_options.gamma = sigma          % kernel bandwidth: rbf only
                
                % init pseudo-label 
                LRModel = train([], sourceY, sparse(sourceX), '-s 0 -c 1');
                LR_Cls = predict(targetY, sparse(targetX), LRModel);
                [accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(LR_Cls, targetY);
                LRClsArray = [];
                LRClsArray = [LRClsArray,LR_Cls];
                fprintf(process_f1_file,'%f,',f_measure);
                fprintf(process_AUC_file,'%f,',AUC);
                
                for t = 1:10
                    [newtrainX, newtestX] = JDA(sourceX,targetX,sourceY,LR_Cls,JDA_options);
                    LRModel = train([], sourceX, sparse(newtrainX), '-s 0 -c 1');
                    LR_Cls = predict(targetY, sparse(newtestX), LRModel);
                    
                    [accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(LR_Cls, targetY);
                    fprintf(process_f1_file,'%f,',f_measure);
                    fprintf(process_AUC_file,'%f,',AUC);
                end
                
                fprintf(process_f1_file,'\n');
                fprintf(process_AUC_file,'\n');
                
                resultStr = [learnerName,',',num2str(0),',',num2str(dim),',',num2str(lambda),',',targetName,',',sourceName,',',modelName,',',num2str(f_measure),',',num2str(AUC)]
                fprintf(file,'%s\n',resultStr);
            end
            
        end
    end
end
