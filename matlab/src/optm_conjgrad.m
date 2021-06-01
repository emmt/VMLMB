% optm_conjgrad - linear conjugate gradient algorithm.
%
% This routine iteratively solves the linear system:
%
%     A.x = b
%
% Arguments:
%    A   - The left-hand-side operator, it is the name or a function handle
%          of a user defined function which takes a single argument and
%          returns operator `A` applied to this argument.
%    b   - The right-hand-side vector of the normal equations.
%    x0  - The initial guess for the solution `x`.
%    tol - The tolerance(s) for convergence, can be one or two values: `rtol`
%           (as if `atol` = 0) or [`rtol`, `atol`] where `atol` and `rtol` are
%           the absolute and relative tolerances.  Convergence occurs when the
%           Euclidean norm of the residuals is less or equal the largest of
%           `atol` and `rtol` times the Eucliden norm of the initial
%           residuals.
%    maxiter - The maximum number of iterations.
%
function x = optm_conjgrad(A, b, x0, tol, maxiter)

    % Check options.
    if length(tol) == 1
        rtol = tol(1);
        atol = 0.0;
    elseif length(tol) == 2
        rtol = tol(1);
        atol = tol(2);
    else
        error("tol must be `rtol` or `[rtol, atol]`");
    end

    % Resolve function handle.
    if ischar(A)
        A = str2func(A);
    end

    % Initialization.
    x = x0;
    r = b - A(x);

    % Conjugate gradient iterations.
    k = 1;
    rho = 0.0;
    while 1
        rhoprev = rho;
        rho = optm_inner(r, r);
        if k == 1
            epsilon = atol + rtol*sqrt(rho);
        end
        if sqrt(rho) <= epsilon
            % Normal convergence.
            break
        elseif k > maxiter
            fprintf("WARNING - too many (%d) conjugate gradient iterations\n", ...
                    maxiter);
            break
        end
        if k == 1
            % First iteration.
            p = r;
        else
            p = r + (rho/rhoprev)*p;
        end
        q = A(p);
        gamma = optm_inner(p, q);
        if gamma <= 0.0
            error("left-hand-side operator A is not positive definite");
            break
        end
        alpha = rho/gamma;
        x = x + alpha*p;
        r = r - alpha*q;
        k = k + 1;
    end
end
