function Run()
global plan;
plan.training = 1;
assert(length(plan.layer) > 1);
input = plan.input;
last_repeat = input.repeat;
start = tic;
for repeat = last_repeat:input.max_repeat
    input.training = 1;
    repeattime = tic;
    incorrect = 0;
    all = 0;
    input.step = 1;
    while (true)
        input.GetImage(1);
        if (input.step == -1)
            break;
        end
        if (mod(input.step, 2) == 0) fprintf('*');end
        ForwardPass();
        BackwardPass();
        incorrect = incorrect + plan.classifier.GetScore();
        all = all + input.batch_size;
        if (mod((input.step - 1), floor(input.train.batches / 5)) == 0)
            fprintf(' number of incorrect train examples = %d, all = %d, err = %.3f%%\n', incorrect, all, incorrect / all * 100);
        end
    end
    input.repeat = repeat + 1;
    fprintf('\nEpoch took = %.1f min.\n', toc(repeattime) / 60);   
    [incr_test, all_test] = Test(0);
    fprintf('\nepoch = %d, number of incorrrent test examples = %d, all = %d, err = %.3f%%\n', repeat, incr_test, all_test, incr_test / all_test * 100);
    save_plan();
end
fprintf('Training is finished. Total time = %.2f mins.\n', toc(start) / 60);
end

function save_plan()
    global plan
    fname = sprintf('models/%s.mat', plan.name);
    fprintf('Saving model to the file : %s\n', fname);
    train = plan.input.train;
    test = plan.input.test;
    val = plan.input.val;
    plan.input.train = [];
    plan.input.test = [];
    plan.input.val = [];    
    save(fname, 'plan');
    plan.input.train = train;
    plan.input.test = test;
    plan.input.val = val;        
end


