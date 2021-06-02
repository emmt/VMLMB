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
%%     [stp, d] = optm_apply_lbfgs(lbfgs, g, sel);
%%
%% where `sel` is a logical array which is true where variables are not blocked
%% by constraints and false elsewhere.
%%
%% See also `optm_new_lbfgs`, `optm_reset_lbfgs`, and `optm_update_lbfgs`.
%%
function [stp, d] = optm_apply_lbfgs(lbfgs, d, sel)
    if nargin < 2 || nargin > 3
        print_usage;
    end
    if (nargin < 2) && (nargin > 3)
        error("invalid number of arguments");
    end
    S = lbfgs.S;
    Y = lbfgs.Y;
    m = lbfgs.m;
    off = lbfgs.mrk + m;
    mp = lbfgs.mp;
    alpha = zeros(m, 1);
    if nargin == 2
        regular = true;
    elseif isempty(sel)
        regular = true;
    elseif sel % true if all elements of sel and true
        regular = true;
    else
        regular = false;
    end
    if regular
        %% Apply the 2-loop L-BFGS recursion algorithm by Matthies & Strang.
        rho = lbfgs.rho;
        gamma = lbfgs.gamma;
        if gamma > 0
            for j = 1:mp
                i = mod((off - j), m) + 1;
                alpha_i = optm_inner(d, S{i})/rho(i);
                d -= alpha_i*(Y{i});
                alpha(i) = alpha_i;
            end
            if gamma != 1
                d *= gamma;
            end
            for j = mp:-1:1
                i = mod((off - j), m) + 1;
                alpha_i = alpha(i);
                beta = optm_inner(d, Y{i})/rho(i);
                d += (alpha_i - beta)*(S{i});
            end
        end
    else
        %% Apply L-BFGS recursion on a subset of free variables specified by a
        %% boolean array.
        rho = zeros(m, 1);
        gamma = 0.0;
        last_i = -1;
        full_size = size(d);
        z = d(sel);
        d = []; % free some memory?
        for j = 1:mp
            i = mod((off - j), m) + 1;
            s_i = S{i}(sel);
            y_i = Y{i}(sel);
            rho_i = optm_inner(s_i, y_i);
            if rho_i > 0
                if gamma <= 0
                    gamma = rho_i/optm_inner(y_i, y_i);
                end
                alpha_i = optm_inner(z, s_i)/rho_i;
                z -= alpha_i*y_i;
                alpha(i) = alpha_i;
                rho(i) = rho_i;
                last_i = i;
            end
        end
        if gamma > 0 && gamma != 1
            z *= gamma;
        end
        for j = mp:-1:1
            i = mod((off - j), m) + 1;
            rho_i = rho(i);
            if rho_i > 0
                if (i != last_i)
                    s_i = S{i}(sel);
                    y_i = Y{i}(sel);
                end
                alpha_i = alpha(i);
                beta = optm_inner(z, y_i)/rho_i;
                z += (alpha_i - beta)*s_i;
                last_i = i;
            end
        end
        s_i = []; % free some memory?
        y_i = []; % free some memory?
        d = zeros(full_size);
        d(sel) = z;
    end
    if gamma > 0
        stp = 1.0;
    else
        stp = 0.0;
    end
end
