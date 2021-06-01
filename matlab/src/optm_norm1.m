%%    norm = optm_norm1(x);
%%
%% yields the L1 norm of `x` that is `sum(abs(x(:)))` but computed as
%% efficiently as possible.
%%
%% See also `optm_inner`, `optm_norm2`, and `optm_norminf`.
%%
function norm = optm_norm1(x)
    x = x(:);
    norm = sum(abs(x));
end
