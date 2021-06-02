%%    norm = optm_norminf(x);
%%
%% yields the infinite norm of `x` that is `max(abs(x(:)))` but computed as
%% efficiently as possible.
%%
%% See also `optm_inner`, `optm_norm1`, and `optm_norm2`.
function norm = optm_norminf(x)
    x = x(:);
    norm = max(abs(x));
end
