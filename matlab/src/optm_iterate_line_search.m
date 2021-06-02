%% optm_iterate_line_search - Make next line-search iteration.
%%
%% This function shall be called as:
%%
%%     lnsrch = optm_iterate_line_search(lnsrch, phi(lnsrch.step));
%%
%% to update line-search context `lnsrch` accounting for the function value
%% `phi(lnsrch.step)` at the current step `lnsrch.step`.  To find the step
%% along a search direction `d` starting at `x0` and for the multi-variate
%% function `f(x)`, the function `phi` shall be defined as:
%%
%%     phi(alpha) = f(x0 + alpha*d)
%%
%% Upon return of `optm_iterate_line_search`, the line-search context `lnsrch`
%% is updated so that `lnsrch.state` is normally one of:
%%
%% 1: Line-search in progress.  The next step to try is `lnsrch.step`,
%%    `optm_iterate_line_search` shall be called with the new function value at
%%    `x0 + lnsrch.step*d`.
%%
%% 2: Line-search has converged.  The step length `lnsrch.step` is left
%%    unchanged and `x0 + lnsrch.step*d` is the new iterate of the optimization
%%    method.
%%
%% Upon creation of the line-search instance, `lnsrch.state` is set to 0.  A
%% strictly negative value for `lnsrch.state` may be used to indicate an error.
%% A typical usage is:
%%
%%     lnsrch = optm_new_line_search();
%%     x = ...;   % initial solution
%%     fx = f(x); % objective function
%%     while (true)
%%         gx = grad_f(x);
%%         if (optm_norm2(gx) <= gtol)
%%             break; % a solution has been found
%%         end
%%         d = next_search_direction(...);
%%         stp = guess_initial_step_size(...);
%%         f0 = fx;
%%         x0 = x;
%%         df0 = optm_inner(d, gx);
%%         lnsrch = optm_start_line_search(lnsrch, f0, df0, stp);
%%         while (lnsrch.state == 1)
%%             x = x0 + lnsrch.step*d;
%%             fx = f(x);
%%             lnsrch = optm_iterate_line_search(lnsrch, fx);
%%         end
%%     end
function lnsrch = optm_iterate_line_search(lnsrch, f)
    finit = lnsrch.finit;
    ginit = lnsrch.ginit;
    step  = lnsrch.step;
    ftol  = lnsrch.ftol;
    if (f <= finit + ftol*(ginit*step))
        %% Linesearch has converged.
        lnsrch.state = 2;
    else
        %% Linesearch has not converged.
        smin = lnsrch.smin;
        smax = lnsrch.smax;
        if (smin < smax)
            %% Try a safeguarded parabolic interpolation step.
            q = -ginit*step;
            r = 2*((f - finit) + q);
            if (q <= smin*r)
                gamma = smin;
            elseif (q >= smax*r)
                gamma = smax;
            else
                gamma = q/r;
            end
        elseif (smin == smax)
            gamma = smin;
        else
            error("invalid fields smin and smax")
        end
        lnsrch.step = gamma*step;
        lnsrch.state = 1;
    end
end
