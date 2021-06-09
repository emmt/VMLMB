%%    xnorm = optm_norminf(x);
%%
%% yields the infinite norm of `x` that is `max(abs(x(:)))` but computed as
%% efficiently as possible.
%%
%% See also `optm_inner`, `optm_norm1`, and `optm_norm2`.
function xnorm = optm_norminf(x)
    xnorm = norm(x(:), Inf); % <-- the fastest in this context
    %%xnorm = max(abs(x(:)));
    %%xnorm = max(max(x(:)), -min(x(:)));
end
