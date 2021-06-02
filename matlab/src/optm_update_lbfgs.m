%%     [lbfgs, accept] = optm_update_lbfgs(lbfgs, s, y)
%%
%% updates the L-BFGS model of the Hessian stored by `lbfgs` with `s` the
%% change in variables to memorize and `y` the corresponding change in
%% gradient.  The result is the updated context and a boolean value `accept`
%% indicating whether `s` and `y` were suitable to update an L-BFGS
%% approximation that be positive definite.
%%
%% Also see `optm_new_lbfgs`, `optm_reset_lbfgs`, and `optm_apply_lbfgs`.
function [lbfgs, accept] = optm_update_lbfgs(lbfgs, s, y)
    if nargin != 3
        print_usage;
    end
    m = lbfgs.m;
    accept = false;
    if m >= 1
        sty = optm_inner(s, y);
        if (sty > 0)
            accept = true;
            mrk = mod(lbfgs.mrk, m) + 1;
            lbfgs.S{mrk} = s;
            lbfgs.Y{mrk} = y;
            lbfgs.rho(mrk) = sty;
            lbfgs.mrk = mrk;
            lbfgs.mp = min(lbfgs.mp + 1, m);
            lbfgs.gamma = sty/optm_inner(y, y);
        end
    end
end
