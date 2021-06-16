%%     [x, status] = optm_conjgrad(A, b, x0, ...)
%%
%% runs the (preconditioned) linear conjugate gradient algorithm to solve the
%% system of equations `A*x = b` in `x`.
%%
%% Argument `x0` specifies the initial solution.  If `x0` is an empty array,
%% i.e. `[]`, `x` is initially an array of zeros.
%%
%% Argument `A` implements the *left-hand-side (LHS) matrix* of the equations.
%% It may be a function name or handle and is called as `A(x)` to compute the
%% result of `A*x`.
%%
%% Note that, as `A` and the preconditioned `M` must be symmetric, it may be
%% faster to apply their adjoint.
%%
%% Argument `b` is the *right-hand-side (RHS) vector* of the equations.
%%
%% Algorithm parameters can be specified as name-value pair of arguments.
%% Possible parameters are:
%%
%% * 'precond' is to specify a preconditioner `M`.  It may be a function name
%%   or handle and is called as `M(x)` to compute the result of `M*x`.  By
%%   default, the un-preconditioned version of the algorithm is run.
%%
%% * 'maxiter' is to specify the maximum number of iterations to perform which
%%   is `intmax()` by default.
%%
%% * 'restart' is to specify the number of consecutive iterations before
%%   restarting the conjugate gradient recurrence.  Restarting the algorithm is
%%   to cope with the accumulation of rounding errors.  By default, `restart =
%%   min(50,numel(x)+1)`.  Set `restart` to a value less or equal zero or
%%   greater than `maxiter` if you do not want that any restarts ever occur.
%%
%% * 'ftol' is to specify the absolute and relative tolerances for the function
%%   reduction as `[fatol,frtol]` or just `frtol` to assume `fatol = 0`.  By
%%   default, `ftol = [0.0,1e-8]`.
%%
%% * 'gtol' is to specify the absolute and relative tolerances for stopping the
%%   algorithm based on the gradient of the objective function as
%%   `[gatol,grtol]` or just `grtol` to assume `gatol = 0`.  Convergence occurs
%%   when the Mahalanobis norm of the residuals (which is that of the gradient
%%   of the associated objective function) is less or equal the largest of
%%   `gatol` and `grtol` times the Mahalanobis norm of the initial residuals.
%%   By default, `gtol = [0.0,1e-5]`.
%%
%% * 'xtol' is to specify the absolute and relative tolerances for the change
%%   in variables as `[xatol,xrtol]` or just `xrtol` to assume `xatol = 0`.  By
%%   default, `xtol = [0.0,1e-6]`.
%%
%% * 'verb' is to specify whether to to print various information at each
%%   iterations.
%%
%%
%% ## Convergence criteria
%%
%% Provided `A` be positive definite, the solution `x` of the equations
%% `A*x = b` is unique and is also the minimum of the following convex
%% quadratic objective function:
%%
%%     f(x) = (1/2)*x'*A*x - b'*x + ϵ
%%
%% where `ϵ` is an arbitrary constant.  The gradient of this objective
%% function is:
%%
%%     ∇f(x) = A*x - b
%%
%% hence solving `A*x = b` for `x` yields the minimum of `f(x)`.  The
%% variations of `f(x)` between successive iterations, the norm of the gradient
%% `∇f(x)` or the norm of the variation of variables `x` may be used to decide
%% the convergence of the algorithm (see keywords `ftol`, `gtol` and `xtol`
%% above).
%%
%% Let `x_{k}`, `f_{k} = f(x_{k})` and `∇f_{k} = ∇f(x_{k})` denote the
%% variables, the objective function and its gradient at iteration `k`.  The
%% argument `x` gives the initial variables `x_{0}`.  Starting with `k = 0`,
%% the different possibilities for the convergence of the algorithm are listed
%% below.
%%
%% * The convergence in the function reduction between succesive iterations occurs
%%   at iteration `k ≥ 1` if:
%%
%%       f_{k-1} - f_{k} ≤ max(fatol, frtol*max_{k' ≤ k}(f_{k'-1} - f_{k'}))
%%
%% * The convergence in the gradient norm occurs at iteration `k ≥ 0` if:
%%
%%       ‖∇f_{k}‖_M ≤ max(gatol, grtol*‖∇f_{0}‖_M)
%%
%%   where `‖u‖_M = sqrt(u'*M*u)` is the Mahalanobis norm of `u` with precision
%%   matrix `M` which is equal to the usual Euclidean norm of `u` if ; no
%%   preconditioner is used or if `M` is the identity.
%%
%% * The convergence in the variables occurs at iteration `k ≥ 1` if:
%%
%%       ‖x_{k} - x_{k-1}‖ ≤ max(xatol, xrtol*‖x_{k}‖)
%%
%% In the conjugate gradient algorithm, the objective function is always reduced
%% at each iteration, but be aware that the gradient and the change of variables
%% norms are not always reduced at each iteration.
%%
%%
%% ## Returned Status
%%
%% The function returns the solution `x` and a status code which indicates the
%% reason of the algorithm termination.  A negative status indicates that some
%% error occured, e.g. the left-hand-side matrix `A` or the preconditioner `M`
%% are found to be not positive definite.  The function `optm_reason` may be
%% called to have a textual description of the meaning of the returned status.
function [x, status] = optm_conjgrad(A, b, x, varargin)

    %% Constants.  Calling inf, nan, true or false takes too much time (2.1µs
    %% instead of 0.2µs if stored in a variable), so use local variables to pay
    %% the price only once.
    TRUE = true();
    FALSE = false();

    %% Default settings (all absolute tolerances set to zero).
    maxiter = intmax();
    restart = min(50, numel(b));
    precond = FALSE;
    verb = 0;
    ftol = 1e-8;
    gtol = 1e-5;
    xtol = 1e-6;
    M = [];
    if mod(length(varargin), 2) ~= 0
        error('parameters must be specified as pairs of names and values');
    end
    for i = 1:2:length(varargin)
        key = varargin{i};
        val = varargin{i+1};
        switch key
            case 'ftol'
                ftol = check_tolerance(key, val);
            case 'gtol'
                gtol = check_tolerance(key, val);
            case 'xtol'
                xtol = check_tolerance(key, val);
            case 'maxiter'
                maxiter = val;
            case 'restart'
                restart = val;
            case 'precond'
                M = val;
                if ischar(M)
                    M = str2func(M);
                end
            case 'verb'
                verb = val;
            otherwise
                error('invalid parameter name `%s`', key);
        end
    end

    %% Resolve function handle.
    if ischar(A)
        A = str2func(A);
    end

    %% Starting solution.
    if nargin < 3 || isempty(x)
        x = zeros(size(b));
        x_is_zero = TRUE;
    else
        x_is_zero = (optm_norm2(x) == 0.0);
    end

    %% Initialize local variables.
    mesg = 0;
    rho = 0.0;
    phi = 0.0;
    phimax = 0.0;
    xtest = any(xtol > 0);
    if verb > 0
        time = @() 86400E3*now(); % yields number of milliseconds
        t0 = time();
    end

    %% Conjugate gradient iterations.
    k = 0;
    while TRUE
        %% Is this the initial or a restarted iteration?
        restarting = (k == 0 || (restart > 0 && mod(k, restart) == 0));

        %% Compute residuals and their squared norm.
        if restarting
            %% Compute residuals.
            if x_is_zero
                %% Spare applying A since x = 0.
                r = b;
                x_is_zero = FALSE;
            else
                %% Compute r = b - A*x.
                r = b - A(x);
            end
        else
            %% Update residuals.
            r = r - alpha*q;
        end
        if precond
            %% Apply preconditioner.
            z = M(r);
        else
            z = r;
        end
        oldrho = rho;
        rho = optm_inner(r, z); % rho = ‖r‖_M^2
        if k == 0
            gtest = optm_tolerance(sqrt(rho), gtol);
        end
        if verb > 0 && mod(k, verb) == 0
            printer(k, time() - t0, phi, rho, r, precond);
        end
        if sqrt(rho) <= gtest
            %% Normal convergence in the gradient norm.
            if verb > 0
                mesg = 'Convergence in the gradient norm.';
            end
            status = optm_status('GTEST_SATISFIED');
            break
        end
        if k >= maxiter
            if verb > 0
                mesg = 'Too many iteration(s).';
            end
            status = optm_status('TOO_MANY_ITERATIONS');
            break
        end

        %% Compute search direction.
        if restarting
            %% Restarting or first iteration.
            p = z;
        else
            %% Apply recurrence.
            beta = rho/oldrho;
            p = z + beta*p;
        end

        %% Compute optimal step size.
        q = A(p);
        gamma = optm_inner(p, q);
        if ~(gamma > 0)
            if verb > 0
                mesg = 'Operator is not positive definite.';
            end
            status = optm_status('NOT_POSITIVE_DEFINITE');
            break
        end
        alpha = rho/gamma;

        %% Update variables and check for convergence.
        x = x + alpha*p;
        phi = alpha*rho/2;  %% phi = f(x_{k}) - f(x_{k+1}) ≥ 0
        phimax = max(phi, phimax);
        if phi <= optm_tolerance(phimax, ftol)
            %% Normal convergence in the function reduction.
            if verb > 0
                mesg = 'Convergence in the function reduction.';
            end
            status = optm_status('FTEST_SATISFIED');
            break
        end
        if xtest && alpha*optm_norm2(p) <= optm_tolerance(x, xtol)
            %% Normal convergence in the variables.
            if verb > 0
                mesg = 'Convergence in the variables.';
            end
            status = optm_status('XTEST_SATISFIED');
            break
        end

        %% Increment iteration number.
        k = k + 1;
    end
    if verb > 0
        %% Print last iteration if not yet done and termination message.
        if mod(k, verb) ~= 0
            printer(k, time() - t0, phi, rho, r, precond);
        end
        if ischar(mesg)
            fprintf('# %s\n', mesg);
        end
    end
    if status < 0 && nargout < 2
        %% An error occured while the caller does not retrieve the status.
        error(optm_reason(status))
    end
end

function val = check_tolerance(key, val)
    if isscalar(val)
        return
    elseif isvector(val) && numel(val) == 2
        return
    else
        error('parameter `%s` value must be `rtol` or `[atol, rtol]`', key);
    end
end

function printer(k, t, phi, rho, r, precond)
    if precond
        if k == 0
            fprintf('%s\n%s\n', ...
                    '# Iter.   Time (ms)     Δf(x)       ‖∇f(x)‖     ‖∇f(x)‖_M', ...
                    '# ---------------------------------------------------------');
        end
        fprintf('%7d %11.3f %12.4e %12.4e %12.4e\n', ...
                k, t, phi, optm_norm2(r), sqrt(rho));
    else
        if k == 0
            fprintf('%s\n%s\n', ...
                    '# Iter.   Time (ms)     Δf(x)       ‖∇f(x)‖', ...
                    '# --------------------------------------------');
        end
        fprintf('%7d %11.3f %12.4e %12.4e\n', ...
                k, t, phi, sqrt(rho));
    end
end
