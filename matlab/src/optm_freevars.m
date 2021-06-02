%% optm_freevars - Get unbounded variables.
%%     msk = optm_freevars(x, xmin, xmax, g);
%%
%% yields a logical mask `msk` of the same size as `x` indicating which
%% elements of `x` are not blocked by the bounds `xmin` and `xmax` when
%% minimizing an objective function whose gradient is `g` at `x`.
%%
%% Empty bounds, that is `xmin = []` or `xmax = []`, are interpreted as
%% unlimited.
%%
%% It is the caller's resposibility to ensure that the bounds are compatible
%% and that the variables are feasible, in other words that `xmin ≤ x ≤ xmax`
%% holds.
%%
%% See also `optm_clamp`.
function msk = optm_freevars(x, xmin, xmax, g)
    if nargin != 4
        print_usage;
    end
    if isempty(xmin)
        if isempty(xmax)
            msk = true(size(x));
        else
            msk = (x < xmax)|(g > 0);
        end
    elseif isempty(xmax)
        msk = (x > xmin)|(g < 0);
    else
        msk = ((x > xmin)|(g < 0))&((x < xmax)|(g > 0));
    end
end
