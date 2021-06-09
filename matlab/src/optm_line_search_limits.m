%%     [amin, amax] = optm_line_search_limits(x0, xmin, xmax, d, dir)
%%
%% Determine the limits `amin` and `amax` for the step length `alpha` in a line
%% search where iterates `x` are given by:
%%
%%     x = proj(x0 ± alpha*d)
%%
%% where `proj(x)` denotes the orthogonal projection on the convex set defined
%% by separable lower and upper bounds `xmin` and `xmax` (unless empty) and
%% where `±` is `-` if `dir` is specified and negative and `+` otherwise.
%%
%% Returned value `amin` is the largest nonnegative step length such that if
%% alpha ≥ amax, then:
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
%% See also `optm_clamp` and `optm_active_variables`.
function [amin, amax] = optm_line_search_limits(x0, xmin, xmax, d, dir)
    if nargin < 4 || nargin > 5
        print_usage;
    end
%    Inf = Inf; % calling Inf takes too much time (2.1µs instead of 0.2µs if
               % stored in a variable), so use a local variable shadowing the
               % function to pay the price once
    %% Quick return if unconstrained.
    if isempty(xmin) && isempty(xmax)
        amin = Inf;
        amax = Inf;
        return
    end
    %% Is `d` an ascent direction?
    if nargin < 5
        ascent = false;
    else
        ascent = (dir < 0);
    end
    amin = Inf;
    amax = 0.0;
    d = d(:); % flatten d (which is assumed to have the same size as x0) for
              % taking the min. and the max.
    if isempty(xmin)
        if ascent
            if max(d) > 0
                amax = Inf;
            end
        else
            if min(d) < 0
                amax = Inf;
            end
        end
    else
        i = [];
        a = [];
        if ascent
            if max(d) > 0
                i = d > 0;
                a = x0 - xmin;
            end
        else
            if min(d) < 0
                i = d < 0;
                a = xmin - x0;
            end
        end
        if ~isempty(a)
            a = a(i) ./ d(i);
            amin = min(amin, min(a));
            amax = max(amax, max(a));
        end
    end
    if isempty(xmax)
        if amax < Inf
            if ascent
                if min(d) < 0
                    amax = Inf;
                end
            else
                if max(d) > 0
                    amax = Inf;
                end
            end
        end
    else
        i = [];
        a = [];
        if ascent
            if min(d) < 0
                i = d < 0;
                a = x0 - xmax;
            end
        else
            if max(d) > 0
                i = d > 0;
                a = xmax - x0;
            end
        end
        if ~isempty(a)
            a = a(i) ./ d(i);
            amin = min(amin, min(a));
            if amax < Inf
                amax = max(amax, max(a));
            end
        end
    end
end
