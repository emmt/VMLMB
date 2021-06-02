%%     alpha = optm_steepest_descent_step(x, d, fx, fmin, delta, gamma);
%%
%% yields the length of the first trial step along the steepest descent
%% direction.  Arguments are:
%%
%% - `x` specifies the current variables (or their Euclidean norm).
%%
%% - `d` is the search direction `d` (or its Euclidean norm) at `x`.  This
%%   direction shall be the gradient (or the projected gradient for a
%%   constrained problem) at `x` up to a change of sign.
%%
%% - `fx` is the value of the objective function at `x`.
%%
%% - `fmin` is an estimate of the minimal value of the objective function.
%%
%% - `delta` is a small step size relative to the norm of the variables.
%%
%% - `gamma` is an estimate of the magnitude of the eigenvalues of the inverse
%%   Hessian of the objective function.
%%
function alpha = optm_steepest_descent_step(x, d, fx, fmin, delta, gamma)
    if nargin != 6
        print_usage;
    end
    if fx > fmin
        %% For a quadratic objective function, the minimum is such that:
        %%
        %%     fmin = f(x) - (1/2)*alpha*d'*∇f(x)
        %%
        %% with `alpha*d = -∇²f(x)\∇f(x)` the Newton step.  Hence:
        %%
        %%     alpha = 2*(f(x) - fmin)/(d'*∇f(x)) = 2*(f(x) - fmin)/‖d‖²
        %%
        %% is an estimate of the step size along `d = -∇f(x)` or the projected
        %% gradient.
        dnorm = norm_of(d);
        alpha = 2*(fx - fmin)/dnorm^2;
        if isfinite(alpha) && alpha > 0
            return
        end
    else
        dnorm = -1;
    end
    if delta > 0 && delta < 1
        if dnorm < 0
            dnorm = norm_of(d);
        end
        xnorm = norm_of(x);
        alpha = delta*xnorm/dnorm;
        if isfinite(alpha) && alpha > 0
            return
        end
    end
    if isfinite(gamma) && gamma > 0
        alpha = gamma;
    else
        if dnorm < 0
            dnorm = norm_of(d);
        end
        alpha = 1.0/dnorm;
    end
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
