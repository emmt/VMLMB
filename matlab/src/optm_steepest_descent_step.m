%%     alpha = optm_steepest_descent_step(x, d, fx, fmin, delta, lambda);
%%
%% yields the length of the first trial step along the steepest descent
%% direction.  Arguments are:
%%
%% - `x` the current variables (or their Euclidean norm).
%%
%% - `d` the search direction `d` (or its Euclidean norm) at `x`.  This
%%   direction shall be the gradient (or the projected gradient for a
%%   constrained problem) at `x` up to a change of sign.
%%
%% - `fx` the value of the objective function at `x`.
%%
%% - `fmin` an estimate of the minimal value of the objective function.
%%
%% - `delta` a small step size relative to the norm of the variables.
%%
%% - `lambda` an estimate of the magnitude of the eigenvalues of the
%%   Hessian of the objective function.
%%
function alpha = optm_steepest_descent_step(x, d, fx, fmin, delta, lambda)
    if nargin != 6
        print_usage;
    end
    # Note that we rely on the behavior of NaN's (in particular for
    # comparisons) to simplify the checking of the validity of the different
    # cases.
    if fx > fmin
        %% For a quadratic objective function, the minimum is such that:
        %%
        %%     fmin = f(x) - (1/2)*alpha*d'*∇f(x)
        %%
        %% with `alpha` the optimal step.  Hence:
        %%
        %%     alpha = 2*(f(x) - fmin)/(d'*∇f(x)) = 2*(f(x) - fmin)/‖d‖²
        %%
        %% is an estimate of the step size along `d` if it is plus or minus the
        %% (projected) gradient.
        dnorm = norm_of(d);
        alpha = 2*(fx - fmin)/dnorm^2;
        if isfinite(alpha) && alpha > 0
            return
        end
    else
        dnorm = -1;
    end
    if delta > 0 && delta < 1
        %% Use the specified small relative step size.
       if dnorm < 0
            dnorm = norm_of(d);
        end
        xnorm = norm_of(x);
        alpha = delta*xnorm/dnorm;
        if isfinite(alpha) && alpha > 0
            return
        end
    end
    %% Use typical Hessian eigenvalue if suitable.
    alpha = 1/lambda;
    if isfinite(alpha) && alpha > 0
        return
    end
    %% Eventually use 1/‖d‖.
    if dnorm < 0
        dnorm = norm_of(d);
    end
    alpha = 1/dnorm;
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
