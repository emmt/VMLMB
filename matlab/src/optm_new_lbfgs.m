%%     lbfgs = optm_new_lbfgs(m);
%%
%% yields a new L-BFGS context capable of memorizing `m` previous steps for
%% modeling the inverse Hessian of a differentiable objective function.
%%
%% Also see `optm_reset_lbfgs`, `optm_update_lbfgs` and `optm_apply_lbfgs`.
function lbfgs = optm_new_lbfgs(m)
    time = @() 86400000*now();
    if nargin ~= 1
        print_usage;
    end
    if m < 0
        error('invalid number of steps to memorize')
    end
    lbfgs.m = m;
    lbfgs.mp = 0;
    lbfgs.mrk = 0;
    lbfgs.gamma = 0.0;
    if m > 0
        lbfgs.rho = zeros(m, 1);
        for i = 1:m
            lbfgs.S{i} = [];
            lbfgs.Y{i} = [];
        end
    else
        lbfgs.rho = [];
        for i = 1:m
            lbfgs.S = {};
            lbfgs.Y = {};
        end
    end
end
