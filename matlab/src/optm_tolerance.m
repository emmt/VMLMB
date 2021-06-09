%%    val = optm_tolerance(arg, tol);
%%
%% Compute a nonnegative value based on argument `arg` and on tolerance
%% settings `tol`.  If `arg` is a scalar, the result is given by:
%%
%%    val = max(0.0, atol, rtol*abs(arg));
%%
%% otherwise, `arg` should be an array and the result is given by:
%%
%%    val = max(0.0, atol, rtol*optm_norm2(arg));
%%
%% The absolute and relative tolerances settings `atol` and `rtol` are
%% specified by `tol = [atol, rtol]` or by `tol = rtol` and `atol = 0` is
%% assumed.
%%
%% If possible, computing the norm of `arg` is avoided (i.e., if `rtol ≤ 0`).
function val = optm_tolerance(arg, tol)
    if isscalar(tol)
        val = 0.0;
        rtol = tol;
    else
        val = max(0.0, tol(1));
        rtol = tol(2);
    end
    if rtol > 0
        if isscalar(arg)
            val = max(val, rtol*abs(arg));
        else
            val = max(val, rtol*optm_norm2(arg));
        end
    end
end
