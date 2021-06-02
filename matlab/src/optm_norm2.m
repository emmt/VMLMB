%%    norm = optm_norm2(x);
%%
%% yields the Euclidean (L2) norm of `x` that is `sqrt(sum(x(:).^2))` but
%% computed as efficiently as possible.
%%
%% See also `optm_inner`, `optm_norm1`, and `optm_norminf`.
function norm = optm_norm2(x)
    x = x(:);
    norm = sqrt(sum(x.*x));
end
