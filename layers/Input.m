classdef Input < Layer
    properties
        repeat
        batch_size
        step
        max_repeat                 
        file_pattern
        meanX
        train
        test
    end
    methods
        function obj = Input(json)
            obj@Layer(json);
            global plan            
            obj.repeat = Val(plan.all_uploaded_weights, 'plan.input.repeat', 1);
            obj.step = 1;            
            obj.batch_size = Val(json, 'batch_size', 128);            
            obj.max_repeat = 10000;                     
            obj.file_pattern = Val(json, 'file_pattern', '.');
            try
                tmp = load(sprintf('%s/meta.mat', obj.file_pattern));
                obj.meanX = tmp.meanX;
            catch
            end
            [obj.train.X, obj.train.Y, obj.train.batches] = obj.LoadData([obj.file_pattern, 'train.mat']);
            [obj.test.X, obj.test.Y, obj.test.batches] = obj.LoadData([obj.file_pattern, 'test.mat']);
            obj.gpu.vars.out = plan.GetGID();
            obj.gpu.vars.Y = plan.GetGID();         
            plan.input = obj;       
            obj.Finalize();   
        end       
        
        function FPgpu(obj) 
        end 
        
        function FP(obj) 
        end
        
        function X = Process(obj, X_)
            if ndims(X_) == 3 
                X = zeros([obj.batch_size, obj.dims]);
                X(1, :, :, :) = X_(1:obj.dims(1), 1:obj.dims(2), :);
            elseif ndims(X_) == 4
                if size(X_, 2) == obj.dims(1) && size(X_, 3) == obj.dims(2)
                    X = X_;
                else
                    X = zeros([obj.batch_size, obj.dims]);
                    X(:, :, :, :) = X_(:, 1:obj.dims(1), 1:obj.dims(2), :);
                end
            end            
            X = X - repmat(reshape(obj.meanX, [1, obj.dims(1), obj.dims(2), obj.dims(3)]), [obj.batch_size, 1, 1, 1]);            
        end
        
        function GetImage(obj, train)        
            global plan
            if isempty(obj.train.X)
                [X, Y, obj.step] = RetriveImageRaw(obj, obj.step, train);
            else
                [X, Y, obj.step] = RetriveImageMat(obj, obj.step, train);
            end
            obj.cpu.vars.out = X;
            obj.cpu.vars.Y = Y;
            if (obj.on_gpu)
                C_(CopyToGPU, obj.gpu.vars.out, single(obj.cpu.vars.out(:, :)));
                C_(CopyToGPU, obj.gpu.vars.Y, single(obj.cpu.vars.Y(:, :)));                
            end
        end
        
        function [X, Y, step] = RetriveImageRaw(obj, step, train)                         
            X = zeros(obj.batch_size, obj.dims(1), obj.dims(2), 3);
            Y = zeros(obj.batch_size, 1000);
            from = ((step - 1) * obj.batch_size + 1);
            to = from + obj.batch_size - 1;
            for i = from : to
                name = sprintf('%s/ILSVRC2012_val_%s.JPEG', obj.file_pattern, sprintf('%08d', i));
                idx = i - from + 1;
                img = single(imread(name));
                X(idx, :, :, :) = img(1:obj.dims(1), 1:obj.dims(2), :);
                Y(idx, obj.Y(i)) = 1;
            end
            X = obj.Process(X);
            step = step + 1;
        end                
        
        
        function [X, Y, batches] = LoadData(obj, file_pattern)
            X = [];
            Y = [];
            batches = [];
            if (strcmp(file_pattern(1:4), 'http'))                
                idxs = strfind(file_pattern, '/');                
                dir_path = sprintf('data%s', file_pattern(idxs(end - 1):idxs(end)));
                if ~exist(dir_path, 'file')
                    mkdir(dir_path);
                end
                target = sprintf('data%s', file_pattern(idxs(end - 1):end));   
                download(sprintf('%s', file_pattern), sprintf('%s', target));                
            else
                target = file_pattern;
            end
            if exist(target, 'file')
                load(target, 'X', 'Y', 'batches');
                obj.batch_size = size(X, 1);
            end
        end
        
        function [X, Y, step] = RetriveImageMat(obj, step, train)
            X = [];
            Y = [];
            if (train == 1)
                source = obj.train;
            elseif (train == 0) 
                source = obj.test;
            end
            if (step > source.batches)
                step = -1;
                return;
            end
            X = source.X{step};
            Y = source.Y{step};            
            step = step + 1;
        end                  
    end
end


