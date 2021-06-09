%%     xp = optm_clamp(x, xmin, xmax);
%%
%% yields `x` restricted to the range `[xmin,xmax]` element-wise.  Empty
%% bounds, that is `xmin = []` or `xmax = []`, are interpreted as unlimited.
%%
%% It is the caller's resposibility to ensure that the bounds are compatible,
%% in other words that `xmin ≤ xmax` holds.
%%
%% See also `optm_active_variables` and `optm_line_search_limits`.
function x = optm_clamp(x, xmin, xmax)
    if nargin ~= 3
        print_usage;
    end
    if ~isempty(xmin)
        x = bsxfun(@max, x, xmin);
    end
    if ~isempty(xmax)
        x = bsxfun(@min, x, xmax);
    end
end
