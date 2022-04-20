%%     [amin, amax] = optm_line_search_limits(x0, xmin, xmax, pm, d)
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
    amin = INF;
    if isempty(xmin) && isempty(xmax)
        %% Quick return if unconstrained.
        amax = INF;
        return
    end
    amax = -INF; % Upper step length bound not yet found.
    backward = (pm < 0); % Are we moving in backward direction?
    dmin = min(d(:));
    dmax = max(d(:));
    if isempty(xmin)
        %% No lower bound set.
        if backward
            if dmax > 0
                amax = INF;
            end
        else
            if dmin < 0
                amax = INF;
            end
        end
    else
        %% Find step sizes to reach any lower bounds.
        a = [];
        if backward
            if dmax > 0
                i = d > 0;
                a = (x0 - xmin)(i) ./ d(i);
            end
        else
            if dmin < 0
                i = d < 0;
                a = (xmin - x0)(i) ./ d(i);
            end
        end
        if ~isempty(a)
            amin = min(a);
            amax = max(a);
        end
    end
    if isempty(xmax)
        %% No upper bound set.
        if amax < INF
            if backward
                if dmin < 0
                    amax = INF;
                end
            else
                if dmax > 0
                    amax = INF;
                end
            end
        end
    else
        %% Find step sizes to reach any upper bounds.
        a = [];
        if backward
            if dmin < 0
                i = d < 0;
                a = (x0 - xmax)(i) ./ d(i);
            end
        else
            if dmax > 0
                i = d > 0;
                a = (xmax - x0)(i) ./ d(i);
            end
        end
        if ~isempty(a)
            amin = min(amin, min(a));
            if amax < INF
                amax = max(amax, max(a));
            end
        end
    end
    if amax < 0
        amax = INF;
    end
end
