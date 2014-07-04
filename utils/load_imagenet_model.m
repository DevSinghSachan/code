function load_imagenet_model(batch_size)    
    global plan
    if (~exist('batch_size', 'var'))
        batch_size = 128;
    end            
    if (exist('plan', 'var') ~= 1) || (isempty(plan))
        json = ParseJSON(sprintf('plans/imagenet.txt'));
        json{1}.batch_size = batch_size;
        Plan(json, '/Users/wojto/data_tmp/imagenet.mat');
        plan.input.step = 1;
    end
end
