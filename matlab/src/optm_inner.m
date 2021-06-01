%% optm_inner - Compute the generalized inner product of its arguments.
%%
%% Arguments must have the same sizes and be real-valued (not complex).
%%
function s = optm_inner(x, y)
    s = sum(bsxfun(@times, x(:), y(:)));
end
