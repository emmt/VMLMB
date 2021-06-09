%%    xnorm = optm_norm2(x);
%%
%% yields the Euclidean (L2) norm of `x` that is `sqrt(sum(x(:).^2))` but
%% computed as efficiently as possible.  `x` must be a real-valued array.
%%
%% See also `optm_inner`, `optm_norm1`, and `optm_norminf`.
function xnorm = optm_norm2(x)
    %%xnorm = norm(x(:), 2);
    %%xnorm = sqrt(x(:)'*x(:)); % <---- the fastest in an inlined function
    %%xnorm = sqrt(sum(x(:).*x(:)));
    xnorm = sqrt(sum((x.*x)(:))); % <-- the fastest in this context
end
