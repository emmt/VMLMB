%%     amax = optm_line_search_step_max(x0, xmin, xmax, pm, d);
%%
%% Determine the step size `amax` along search direction `d` such that:
%%
%%      alpha ≥ amax  ==>  proj(x0 ± alpha*d) = proj(x0 ± amax*d)
%%
%% where `proj(x)` denotes the orthogonal projection on the convex set defined
%% by separable lower and upper bounds `xmin` and `xmax` (unless empty) and
%% where `±` is `-` if `pm` is negative `+` otherwise.
%%
%% Restrictions: `x0` must be feasible and must have the same size as `d`; this
%% is not verified for efficiency reasons.
%%
%% See also: `optm_clamp`, `optm_unblocked_variables`, and
%% `optm_line_search_limits`.
function amax = optm_line_search_step_max(x0, xmin, xmax, pm, d)
    if nargin ~= 5
        print_usage;
    end
    INF = Inf();
    unbounded_below = isempty(xmin);
    unbounded_above = isempty(xmax);
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
                amax = max((x0 - xmin)(i) ./ d(i));
            end
        end
        if amax < INF && min(d(:)) < 0
            if unbounded_above
                amax = INF;
            else
                i = d < 0;
                amax = max(amax, max((x0 - xmax)(i) ./ d(i)));
            end
        end
    else
        %% We are moving in the forward direction.
        if min(d(:)) < 0
            if unbounded_below
                amax = INF;
            else
                i = d < 0;
                amax = max((xmin - x0)(i) ./ d(i));
            end
        end
        if amax < INF && max(d(:)) > 0
            if unbounded_above
                amax = INF;
            else
                i = d > 0;
                amax = max(amax, max((xmax - x0)(i) ./ d(i)));
            end
        end
    end
    if amax < 0
        amax = INF;
    end
end
