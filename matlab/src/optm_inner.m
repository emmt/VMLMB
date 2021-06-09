%%     s = optm_inner(x, y)
%%
%% Compute the generalized inner product of `x` and `y`. Arguments must have
%% the same sizes and be real-valued (not complex).
%%
%% If `x` and `y` are the same, it is more efficient to call `optm_norm2(x)^2`.
function s = optm_inner(x, y)
    s = x(:)'*y(:); % <-- the fastest when x and y are not the same
    %%s = sum(x(:).*y(:));
    %%s = sum(bsxfun(@times, x(:), y(:)));
    %%s = sum((x.*y)(:));
    %%s = sum(bsxfun(@times, x, y)(:));
end
