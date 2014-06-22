addpath(genpath('.'));
% on_gpu set to true, only if you built cuda files in
% directory /gpu (by calling make).
on_gpu = false;
% Choose any out of following three plans.
% Plan('cifar');
% Plan('imagenet'); % Train only on GPU. Don't train on CPU.
Plan('mnist', on_gpu);
Run();
