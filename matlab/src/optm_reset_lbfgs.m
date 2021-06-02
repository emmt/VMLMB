%%     lbfgs = optm_reset_lbfgs(lbfgs);
%%
%% resets the L-BFGS model stored in context `lbfgs`, thus forgetting any
%% memorized information.  This also frees most memory allocated by the
%% context.
%%
%% Also see `optm_new_lbfgs`, `optm_update_lbfgs`, and `optm_apply_lbfgs`.
function lbfgs = optm_reset_lbfgs(lbfgs)
    if nargin != 1
        print_usage;
    end
    lbfgs.mp = 0;
    lbfgs.mrk = 0;
    lbfgs.gamma = 0.0;
    m = lbfgs.m;
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
