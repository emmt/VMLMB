%% optm_start_line_search - Initiate a new line-search.
%%
%% This function shall be called as:
%%
%%     lnsrch = optm_start_line_search(lnsrch, f0, df0, stp);
%%
%% to start a new line-search.  The first argument is the line-search context
%% which is updated to reflect that a new line-search has been started.  The
%% two next arguments are the value of the objective function and its
%% directional derivative at the current iterate:
%%
%%     f0 = f(x0);
%%     df0 = optm_inner(d, grad_f(x0));
%%
%% with `f(x)` and `grad_f(x)` the objective function and its gradient, `x0`
%% the variables at the start of the line search and `d` the search direction.
%% The function `optm_inner` yields the inner product of its arguments.  It is
%% assumed that, during the line-search, the variables are given by:
%%
%%    x = x0 + alpha*d;
%%
%% with `alpha` the step length.  Argument `stp > 0` is the first step length
%% to try.
%%
%% Upon return of `optm_start_line_search`, the line-search context `lnsrch` is
%% updated so that `lnsrch.stage` is normally equal to 1 indicating that the
%% next trial step is `lnsrch.step` (which should be equal to `stp` in that
%% case), and that `optm_iterate_line_search` shall be called with the new
%% function value at `x0 + lnsrch.step*d`.
%%
%% See documentation of `optm_iterate_line_search` for a complete usage example.
function lnsrch = optm_start_line_search(lnsrch, f0, df0, stp)
    if isfinite(df0) && df0 < 0
        if ~isfinite(stp) || stp <= 0
            stage = -1;
            stp = 0.0;
            error("first step to try must be strictly greater than 0");
        else
            stage = 1;
        end
    else
        stage = -1;
        stp = 0.0;
        error("not a descent direction");
    end
    lnsrch.finit = f0;
    lnsrch.ginit = df0;
    lnsrch.step  = stp;
    lnsrch.stage = stage;
end
