%%     [stp, d] = optm_apply_lbfgs(lbfgs, g);
%%
%% applies the L-BFGS approximation of the inverse Hessian stored in `lbfgs`
%% to the "vector" `g`.  The result is returned in `d` while `stp` indicates
%% the estimated step length to first try.  If no valid L-BFGS updates were
%% available, `stp = 0` and `d` is equal to `g` (except that `d(i) = 0` if
%% the `i`-th variable is blocked and such constraints apply, see below);
%% otherwise, `stp = 1`.
%%
%% The L-BFGS approximation may be restricted to a subset of "free variables"
%% by specifying an additional argument:
%%
%%     [stp, d] = optm_apply_lbfgs(lbfgs, g, freevars);
%%
%% where `freevars` is a logical array which is true (1) where variables are
%% not blocked by constraints and false (0) elsewhere.
%%
%% See also `optm_new_lbfgs`, `optm_reset_lbfgs`, and `optm_update_lbfgs`.
%%
function [stp, d] = optm_apply_lbfgs(lbfgs, d, freevars)
    if nargin < 2 || nargin > 3
        print_usage;
    end
    if (nargin < 2) && (nargin > 3)
        error('invalid number of arguments');
    end
    S = lbfgs.S;
    Y = lbfgs.Y;
    m = lbfgs.m;
    off = lbfgs.mrk + m;
    mp = lbfgs.mp;
    alpha = zeros(m, 1);
    if nargin == 2
        regular = true;
    elseif isempty(freevars)
        regular = true;
    elseif freevars % true if all elements of freevars are true
        regular = true;
    else
        regular = false;
        msk = double(freevars);
    end
    if regular
        %% Apply the 2-loop L-BFGS recursion algorithm by Matthies & Strang.
        rho = lbfgs.rho;
        gamma = lbfgs.gamma;
        if gamma > 0
            for j = 1:mp
                i = mod((off - j), m) + 1;
                alpha_i = optm_inner(d, S{i})/rho(i);
                d = d - alpha_i*(Y{i});
                alpha(i) = alpha_i;
            end
            if gamma ~= 1
                d = gamma*d;
            end
            for j = mp:-1:1
                i = mod((off - j), m) + 1;
                beta = optm_inner(d, Y{i})/rho(i);
                d = d + (alpha(i) - beta)*(S{i});
            end
        end
    else
        %% Apply L-BFGS recursion on a subset of free variables specified by a
        %% boolean array.
        rho = zeros(m, 1);
        gamma = 0.0;
        d = d .* msk;
        for j = 1:mp
            i = mod((off - j), m) + 1;
            s_i = S{i};
            y_i = Y{i} .* msk;
            rho_i = optm_inner(s_i, y_i);
            if rho_i > 0
                if gamma <= 0
                    gamma = rho_i/optm_inner(y_i, y_i);
                end
                alpha_i = optm_inner(d, s_i)/rho_i;
                d = d - alpha_i*y_i;
                alpha(i) = alpha_i;
                rho(i) = rho_i;
            end
            s_i = []; % free some memory?
            y_i = []; % free some memory?
        end
        if gamma > 0 && gamma ~= 1
            d = gamma*d;
        end
        for j = mp:-1:1
            i = mod((off - j), m) + 1;
            rho_i = rho(i);
            if rho_i > 0
                beta = optm_inner(d, Y{i})/rho_i;
                d = d + (alpha(i) - beta)*(S{i} .* msk);
            end
        end
    end
    if gamma > 0
        stp = 1.0;
    else
        stp = 0.0;
    end
end
