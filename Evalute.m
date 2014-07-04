function predictions = Evalute(img)
    global plan;
    plan.training = 0; % Turns off dropout.
    img = single(imread(img));    
    plan.input.cpu.vars.out = plan.input.Process(img);
    plan.input.cpu.vars.Y = 0;
    ForwardPass();    
    plan.training = 1; % Turns on dropout.    
    predictions = plan.classifier.cpu.vars.pred(1, :);
end
