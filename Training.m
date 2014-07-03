addpath(genpath('.'));
on_gpu = false;
Plan('cifar', on_gpu);
Run();