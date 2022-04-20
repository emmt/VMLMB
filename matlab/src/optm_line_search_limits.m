%%     [amin, amax] = optm_line_search_limits(x0, xmin, xmax, pm, d);
%%
%% Determine the limits `amin` and `amax` for the step length `alpha` in a line
%% search where iterates `x` are given by:
%%
%%     x = proj(x0 ± alpha*d)
%%
%% where `proj(x)` denotes the orthogonal projection on the convex set defined
%% by separable lower and upper bounds `xmin` and `xmax` (unless empty) and
%% where `±` is `-` if `pm` is negative and `+` otherwise.
%%
%% Returned value `amin` is the largest nonnegative step length such that if
%% `alpha ≤ amin`, then:
%%
%%     proj(x0 ± alpha*d) = x0 ± alpha*d
%%
%% Returned value `amax` is the least nonnegative step length such that if
%% alpha ≥ amax, then:
%%
%%     proj(x0 ± alpha*d) = proj(x0 ± amax*d)
%%
%% In other words, no bounds are overcome if `0 ≤ alpha ≤ amin` and the
%% projected variables are all the same for any `alpha` such that
%% `alpha ≥ amax ≥ 0`.
%%
%% Restrictions: `x0` must be feasible and must have the same size as `d`; this
%% is not verified for efficiency reasons.
%%
%% See also `optm_clamp` and `optm_unblocked_variables`.
function [amin, amax] = optm_line_search_limits(x0, xmin, xmax, pm, d)
    if nargin ~= 5
        print_usage;
    end

    %% Constants.  Calling inf, nan, true or false takes too much time (2.1µs
    %% instead of 0.2µs if stored in a variable), so use local variables to pay
    %% the price only once.
    INF = Inf();
    unbounded_below = isempty(xmin);
    unbounded_above = isempty(xmax);
    amin = INF;
    if unbounded_below && unbounded_above
        %% Quick return if unconstrained.
        amax = INF;
        return
    end
    amax = -INF; % Upper step length bound not yet found.
    if pm < 0
        %% We are moving in the backward direction.
        if max(d(:)) > 0
            if unbounded_below
                amax = INF;
            else
                i = d > 0;
                a = (x0 - xmin)(i) ./ d(i);
                amin = min(a);
                amax = max(a);
            end
        end
        if unbounded_above
            if amax < INF && min(d(:)) < 0
                amax = INF;
            end
        elseif min(d(:)) < 0
            i = d < 0;
            a = (x0 - xmax)(i) ./ d(i);
            amin = min(amin, min(a));
            if amax < INF
                amax = max(amax, max(a));
            end
        end
    else
        %% We are moving in the forward direction.
        if min(d(:)) < 0
            if unbounded_below
                amax = INF;
            else
                i = d < 0;
                a = (xmin - x0)(i) ./ d(i);
                amin = min(a);
                amax = max(a);
            end
        end
        if unbounded_above
            if amax < INF && max(d(:)) > 0
                amax = INF;
            end
        elseif max(d(:)) > 0
            i = d > 0;
            a = (xmax - x0)(i) ./ d(i);
            amin = min(amin, min(a));
            if amax < INF
                amax = max(amax, max(a));
            end
        end
    end
    %% Upper step length bound may be unlimited.
    if amax < 0
        amax = INF;
    end
end
