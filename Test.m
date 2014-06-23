function [incr, all] = Test(train)
global plan;
plan.training = 0; % Turns off dropout.
incr = 0;
all = 0;
input = plan.input;
input.step = 1;
fprintf('Testing:\n');
while (true)
    input.GetImage(train);
    if (input.step == -1)
        break;
    end
    if (mod(input.step, 2) == 0) fprintf('*');end
    ForwardPass();    
    incr = incr + plan.classifier.GetScore();
    all = all + plan.input.batch_size;
end
plan.training = 1; % Turns on dropout.
end
