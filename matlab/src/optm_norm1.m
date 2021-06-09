%%    xnorm = optm_norm1(x);
%%
%% yields the L1 norm of `x` that is `sum(abs(x(:)))` but computed as
%% efficiently as possible.
%%
%% See also `optm_inner`, `optm_norm2`, and `optm_norminf`.
function xnorm = optm_norm1(x)
    xnorm = norm(x(:), 1); % <-- the fastest in this context
    %%xnorm = sum(abs(x(:)));
end
