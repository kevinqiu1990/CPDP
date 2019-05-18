% Transfer Feature Learning with Joint Distribution Adaptation.
% M. Long, J. Wang, G. Ding, J. Sun, and P.S. Yu.
% IEEE International Conference on Computer Vision (ICCV), 2013.

% Contact: Mingsheng Long (longmingsheng@gmail.com)

clear all;
load ./matlab.mat


fileList={'EQ','JDT','LC','ML','PDE'};
attributeNum=61;
labelIndex=62;
% fileList={'Apache','Safe','Zxing'};
% attributeNum=26;
% labelIndex=27;
% fileList = {'ant17','log4j12','lucene24'};
% attributeNum=20;
% labelIndex=21;

% Set algorithm parameters
options.k = 10;
options.lambda = 0.1;
options.ker = 'primal';     % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;        % kernel bandwidth: rbf only
T = 10;

result = [];
fid1 = fopen(strcat('../result/JDA-promise-pre.o'),'wt');
for i = 1:length(fileList)
    tgt = char(fileList{i});
    T = eval(tgt);
    Xt = T(:,1:attributeNum);
    Xt = Xt';
    Yt = T(:,labelIndex);
    Yt(Yt==-1)=0;
    
    for j = 1:length(fileList)
        if(j~=i)
            src = char(fileList{j});
            options.data = strcat(src,'_vs_',tgt);
            
            % Prepare data
            S = eval(src);
            Xs = S(:,1:attributeNum);
            Xs = Xs';
            Ys = S(:,labelIndex);
            Ys(Ys==-1)=0;
            
            % LG evaluation
            b = glmfit(Xs', Ys, 'binomial', 'link', 'logit');%逻辑回归方程; b返回逻辑回归方程的系数
            p = glmval(b, Xt', 'logit');%p返回逻辑回归预测值（概率值0到1);
            Cls = round(p);
            [~,~,~,~,~,fmeasure,~] = evaluate(Cls, Yt);
            fprintf(fid1,'LG=%0.4f\n',fmeasure);
            
            % JDA evaluation
            Cls = [];
            Fmeasure = [];
            for t = 1:T
                fprintf('==============================Iteration [%d]==============================\n',t);
                [Z,A] = JDA(Xs,Xt,Ys,Cls,options);
                Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
                Zs = Z(:,1:size(Xs,2));
                Zt = Z(:,size(Xs,2)+1:end);
                
                b = glmfit(Zs', Ys, 'binomial', 'link', 'logit');%逻辑回归方程; b返回逻辑回归方程的系数
                p = glmval(b, Zt', 'logit');%p返回逻辑回归预测值（概率值0到1);
                Cls = round(p);
                [~,~,~,~,~,fmeasure,~] = evaluate(Cls, Yt);
                fprintf('JDA+LG=%0.4f\n',fmeasure);
                Fmeasure = [Fmeasure;fmeasure];
            end

            result = [result;Fmeasure(end)];
            fprintf('\n\n\n');
        end
    end
end
fid = fopen(strcat('../result/JDA-promise.o'),'wt');
fprintf(fid,'%0.4f\n',result);
fclose(fid1);
fclose(fid);