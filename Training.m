addpath(genpath('.'));
on_gpu = false;
Plan('mnist', on_gpu);
Run();