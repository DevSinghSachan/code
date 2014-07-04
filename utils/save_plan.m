function save_plan(plan_orig)
    fname = sprintf('models/%s.mat', plan_orig.name);
    fprintf('Saving model to the file : %s\n', fname);
    plan = Plan();
    plan.all_uploaded_weights = [];
    for i = 1:length(plan_orig.layer)
        classname = class(plan_orig.layer{i});
        plan.layer{i} = feval(classname, struct('type', classname));
        vars = struct();
        if isfield(plan_orig.layer{i}.cpu.vars, 'W')
            vars.W = plan_orig.layer{i}.cpu.vars.W;
        end
        if isfield(plan_orig.layer{i}.cpu.vars, 'B')
            vars.B = plan_orig.layer{i}.cpu.vars.B;
        end        
        plan.layer{i}.cpu = struct('vars', vars);
    end
    plan.input.repeat = plan_orig.input.repeat;
    save(fname, 'plan');    
end
