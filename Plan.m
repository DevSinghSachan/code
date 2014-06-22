classdef Plan < handle
    properties
        jsons
        debug
        stats
        layer
        input
        classifier
        gid
        time
        default_on_gpu
        upload_weights
        all_uploaded_weights
        lr
        momentum
        training        
        name
    end

    methods
        function obj = Plan(param1, default_on_gpu)
            if (ischar(param1))
                jsons = ParseJSON(sprintf('plans/%s.txt', param1));
                obj.name = param1;
            else
                jsons = param1;
                obj.name = 'noname';
            end
            if (exist('default_on_gpu', 'var'))
                obj.default_on_gpu = default_on_gpu;
            else
                obj.default_on_gpu = 0;
            end

            obj.jsons = jsons;
            obj.gid = 0;
            obj.debug = 0;
            randn('seed', 1);
            rand('seed', 1);
            obj.layer = {};
            weights = sprintf('model/%s.mat', obj.name);
            if (exist(weights , 'file'))
                fprintf('Loading weights from %s.\n', weights);
                obj.all_uploaded_weights = load(weights);
            else
                fprintf('Starting training from scratch.\n')
            end
            global plan cuda
            plan = obj;
            cuda = zeros(2, 1);
            obj.stats = struct('total_vars', 0, 'total_learnable_vars', 0, 'total_vars_gpu', 0);
            for i = 1:length(jsons)
                json = jsons{i};
                if strcmp(json.type(), 'Spec') == 0
                    obj.layer{end + 1} = eval(sprintf('%s(json);', json.type()));
                else
                    plan.lr = json.lr;
                    plan.momentum = json.momentum;
                end
            end
            fprintf('Total number of\n\ttotal learnable vars = %d\n\ttotal vars = %d\n\ttotal vars on the gpu = %d\n', obj.stats.total_learnable_vars, obj.stats.total_vars, obj.stats.total_vars_gpu);
            obj.all_uploaded_weights = [];
        end

        function gid = GetGID(obj)
            gid = obj.gid;
            obj.gid = obj.gid + 1;
        end

    end
end
