addpath(genpath('.'));
download('http://www.cs.nyu.edu/~zaremba/models/imagenet.mat', 'models/imagenet.mat');
on_gpu = false;
Plan('imagenet', on_gpu);
predictions = Evalute('data/doggy.jpg');
fprintf('Top-5 predictions:\n');
display(predictions(1:5));
