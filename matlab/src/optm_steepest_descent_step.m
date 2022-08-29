%%     alpha = optm_steepest_descent_step(x, d, fx, f2nd, fmin, ...
%%                                        dxrel, dxabs);
%%
%% yields the length of the first trial step along the steepest descent
%% direction.  The leading arguments are:
%%
%% - `x` the current variables (or their Euclidean norm).
%%
%% - `d` the search direction `d` (or its Euclidean norm) at `x`.  This
%%   direction shall be the gradient (or the projected gradient for a
%%   constrained problem) at `x` up to a change of sign.
%%
%% - `fx` the value of the objective function at `x`.
%%
%% The function returns the first valid step length that is computed according
%% to the following trailing arguments (in the listed order):
%%
%% - `f2nd` a typical value of the second derivatives of the objective
%%   function.
%%
%% - `fmin` an estimate of the minimal value of the objective function.
%%
%% - `dxrel` a small step size relative to the norm of the variables.
%%
%% - `dxabs` an absolute size in the norm of the change of variables.
function alpha = optm_steepest_descent_step(x, d, fx, f2nd, fmin, dxrel, dxabs)
    if nargin ~= 7
        print_usage;
    end
    % Note that we rely on the behavior of NaN's (in particular for
    % comparisons) to simplify the checking of the validity of the different
    % cases.
    if isfinite(f2nd) && f2nd > 0
        %% Use typical value of second derivative of objective function.
        alpha = 1/f2nd;
        if isfinite(alpha) && alpha > 0
            return
        end
    end
    if isfinite(fmin) && fmin < fx
        %% The behavior of a quadratic objective function f(x) along the search
        %% direction d starting at x is given by:
        %%
        %%     f(x + α⋅d) = f(x) + α⋅⟨d,∇f(x)⟩ + (1/2)⋅α²⋅⟨d,∇²f(x)⋅d⟩
        %%
        %% Taking the derivative in α, it can be found that the minimum in α is
        %% for:
        %%
        %%     α_best = -⟨d,∇f(x)⟩/⟨d,∇²f(x)⋅d⟩
        %%
        %% and it is easy to prove that:
        %%
        %%     min_α f(x + α⋅d) = f(x + α_best⋅d)
        %%                      = f(x) + (1/2)⋅α_best⋅⟨d,∇f(x)⟩
        %%
        %% If ⟨d,∇f(x)⟩ = 0, then α_best = 0; otherwise, if an inferior bound
        %% fmin ≤ f(x) (∀ x) is known, then:
        %%
        %%     fmin ≤ min_α f(x + α⋅d) = f(x) + (1/2)⋅α_best⋅⟨d,∇f(x)⟩
        %%
        %% holds and (assuming a convex function along d):
        %%
        %%  • if ⟨d,∇f(x)⟩ > 0, then α_best is negative and such that:
        %%
        %%        α_best ≥ 2⋅(fmin - f(x))/⟨d,∇f(x)⟩
        %%
        %%  • otherwise, ⟨d,∇f(x)⟩ < 0, then α_best is positive and such that:
        %%
        %%        α_best ≤ 2⋅(fmin - f(x))/⟨d,∇f(x)⟩
        %%
        %% This shows that, in any case, 2⋅(fmin - f(x))/⟨d,∇f(x)⟩ is the step
        %% of maximum length. The idea is to take this step and, if needed,
        %% backtrack from there.
        %%
        %% When `d = -∇f(x)` (steepest descent direction), the step to try is
        %% given by:
        %%
        %%     2⋅(f(x) - fmin)/‖d‖²
        %%
        dnorm = norm_of(d);
        alpha = 2*(fx - fmin)/dnorm^2;
        if isfinite(alpha) && alpha > 0
            return
        end
    else
        dnorm = -1;
    end
    if 0 < dxrel && dxrel < 1
        %% Use relative norm of initial change of variables.
       if dnorm < 0
            dnorm = norm_of(d);
        end
        xnorm = norm_of(x);
        alpha = dxrel*xnorm/dnorm;
        if isfinite(alpha) && alpha > 0
            return
        end
    end
    if isfinite(dxabs) && 0 < dxabs
        %% Use absolute norm of initial change of variables.
        if dnorm < 0
            dnorm = norm_of(d);
        end
        alpha = dxabs/dnorm;
        if isfinite(alpha) && alpha > 0
            return
        end
    end
    error('invalid settings for steepest descent step length');
end

%% This function yields the norm of an argument that may be an array or its
%% Euclidean norm.
function norm = norm_of(x)
    if isscalar(x)
        norm = abs(x);
    else
        norm = optm_norm2(x);
    end
end
